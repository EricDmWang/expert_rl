"""
Multi-Agent Policy Gradient (MAPG) with Expert Guidance

This implementation augments MAPG with expert guidance using Jensen-Shannon divergence.
The expert policies are loaded from pre-trained MADQN models at different training stages.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import signal
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Set matplotlib backend to prevent GUI issues
import matplotlib
matplotlib.use('Agg')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expert_rl.gridworld25_env import GridWorld25v0
from mapg_expert_rl.train_mapg import PolicyNetwork, ValueNetwork, MAPGAgent, RolloutBuffer
from madqn.train_madqn import ImprovedDQNNetwork


class ExpertMAPGAgent(MAPGAgent):
    """MAPG Agent with Expert Guidance using Jensen-Shannon Divergence."""
    
    def __init__(self, obs_dim: int, num_actions: int, num_agents: int,
                 lr: float = 1e-5, gamma: float = 0.97, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, value_loss_coef: float = 0.8, entropy_coef: float = 0.01,
                 max_grad_norm: float = 1.0, hidden_dims: List[int] = [128, 128], 
                 device: torch.device = None, expert_guidance_coef: float = 0.1, 
                 expert_beta_0: float = 1.0):
        super().__init__(obs_dim, num_actions, num_agents, lr, gamma, gae_lambda,
                        clip_ratio, value_loss_coef, entropy_coef, max_grad_norm, 
                        hidden_dims, device)
        
        self.expert_guidance_coef = expert_guidance_coef
        self.expert_beta_0 = expert_beta_0
        self.expert_policies = None  # Will be loaded later
        self.hidden_dims = hidden_dims  # Store hidden_dims for expert network creation
        
    def load_expert_policies(self, expert_model_paths: List[str], expert_stage: str = "ep250"):
        """Load expert policies from MADQN checkpoints for a specific stage."""
        self.expert_policies = []
        
        for agent_idx in range(self.num_agents):
            agent_experts = []
            
            # Find the expert path for the specified stage
            target_episode = expert_stage.replace("ep", "")  # Extract episode number
            expert_path = None
            
            for path in expert_model_paths:
                if expert_stage in path:
                    expert_path = Path(path)
                    break
            
            if expert_path is None:
                print(f"Warning: No expert path found for stage {expert_stage}")
                agent_experts.append(None)
            else:
                # Construct the agent-specific path
                agent_path = expert_path.parent / f"checkpoint_agent_{agent_idx}_ep{target_episode}.pt"
                
                if agent_path.exists():
                    checkpoint = torch.load(agent_path, map_location=self.device)
                    
                    # Create expert DQN network (MADQN uses DQN, not policy networks)
                    expert_net = ImprovedDQNNetwork(
                        obs_dim=self.obs_dim,
                        num_actions=self.num_actions,
                        hidden_dims=self.hidden_dims,
                        use_dueling=True,
                        dropout=0.0
                    ).to(self.device)
                    expert_net.load_state_dict(checkpoint['state_dict'])
                    expert_net.eval()  # Set to evaluation mode
                    
                    agent_experts.append(expert_net)
                    print(f"Loaded expert DQN policy for agent {agent_idx} from stage {expert_stage} at {agent_path}")
                else:
                    print(f"Warning: Expert model not found at {agent_path}")
                    agent_experts.append(None)
            
            self.expert_policies.append(agent_experts)
    
    def compute_jensen_shannon_divergence(self, policy_dist: torch.distributions.Categorical, 
                                        expert_dist: torch.distributions.Categorical) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between current and expert policies."""
        # Get probability distributions
        p = policy_dist.probs
        q = expert_dist.probs
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p = p + eps
        q = q + eps
        
        # Normalize to ensure they sum to 1
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        
        # Compute KL divergences
        kl_pq = F.kl_div(p.log(), q, reduction='none').sum(dim=-1)
        kl_qp = F.kl_div(q.log(), p, reduction='none').sum(dim=-1)
        
        # Jensen-Shannon divergence: 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(p.log(), m, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(q.log(), m, reduction='none').sum(dim=-1)
        
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div
    
    def get_expert_action_distribution(self, obs: torch.Tensor, agent_idx: int, expert_idx: int = 0) -> torch.distributions.Categorical:
        """Get action distribution from expert policy."""
        if (self.expert_policies is None or 
            agent_idx >= len(self.expert_policies) or 
            expert_idx >= len(self.expert_policies[agent_idx]) or
            self.expert_policies[agent_idx][expert_idx] is None):
            # Return uniform distribution if no expert available
            return torch.distributions.Categorical(
                torch.ones(obs.shape[0], self.num_actions, device=self.device) / self.num_actions
            )
        
        expert_net = self.expert_policies[agent_idx][expert_idx]
        with torch.no_grad():
            # DQN network outputs Q-values, convert to action probabilities using softmax
            q_values = expert_net(obs)
            # Apply temperature scaling to make distribution more or less peaked
            # Higher temperature = more exploration, lower temperature = more exploitation
            temperature = 2.0  # Increased temperature for smoother expert guidance
            expert_logits = q_values / temperature
            expert_dist = torch.distributions.Categorical(logits=expert_logits)
        
        return expert_dist
    
    def _update_policy_agent_expert(self, agent_idx: int, obs: torch.Tensor, actions: torch.Tensor,
                                  old_log_probs: torch.Tensor, advantages: torch.Tensor,
                                  returns: torch.Tensor, episode: int, expert_idx: int = 0):
        """Update policy for a specific agent with expert guidance."""
        policy_net = self.policy_networks[agent_idx]
        optimizer = self.policy_optimizers[agent_idx]
        scheduler = self.policy_schedulers[agent_idx]
        
        # Get current policy distribution
        policy_dist = policy_net(obs)
        
        # Get new log probabilities
        new_log_probs = policy_dist.log_prob(actions)
        
        # Compute policy loss (PPO)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Add expert guidance term
        expert_dist = self.get_expert_action_distribution(obs, agent_idx, expert_idx)
        js_div = self.compute_jensen_shannon_divergence(policy_dist, expert_dist)
        
        # Compute beta_t = beta_0 / (1 + episode * decay_rate) for smoother decay
        decay_rate = 0.001  # Slower decay for better convergence
        beta_t = self.expert_beta_0 / (1 + episode * decay_rate)
        
        # Expert guidance loss: -beta_t * D_JS(π^i_t || π^e_i)
        expert_guidance_loss = -beta_t * js_div.mean()
        
        # Total policy loss
        total_policy_loss = policy_loss + self.expert_guidance_coef * expert_guidance_loss
        
        # Compute entropy loss
        entropy_loss = -policy_dist.entropy().mean()
        
        # Total loss
        total_loss = total_policy_loss + self.entropy_coef * entropy_loss
        
        # Update
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'expert_guidance_loss': expert_guidance_loss.item(),
            'total_policy_loss': total_policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'js_divergence': js_div.mean().item(),
            'beta_t': beta_t,
            'clipped_ratio': torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio).mean().item()
        }
    
    def update_expert_guidance(self, obs: torch.Tensor, actions: torch.Tensor, 
                             old_log_probs: torch.Tensor, advantages: torch.Tensor,
                             returns: torch.Tensor, episode: int, expert_idx: int = 0):
        """Update all agents with expert guidance."""
        policy_losses = []
        expert_guidance_losses = []
        total_policy_losses = []
        entropy_losses = []
        js_divergences = []
        beta_values = []
        clipped_ratios = []
        
        # Update each agent's policy with expert guidance
        for agent_idx in range(self.num_agents):
            agent_obs = obs[:, agent_idx]
            agent_actions = actions[:, agent_idx]
            agent_old_log_probs = old_log_probs[:, agent_idx]
            agent_advantages = advantages[:, agent_idx]
            agent_returns = returns[:, agent_idx]
            
            result = self._update_policy_agent_expert(
                agent_idx, agent_obs, agent_actions, agent_old_log_probs,
                agent_advantages, agent_returns, episode, expert_idx
            )
            
            policy_losses.append(result['policy_loss'])
            expert_guidance_losses.append(result['expert_guidance_loss'])
            total_policy_losses.append(result['total_policy_loss'])
            entropy_losses.append(result['entropy_loss'])
            js_divergences.append(result['js_divergence'])
            beta_values.append(result['beta_t'])
            clipped_ratios.append(result['clipped_ratio'])
        
        # Update value network (same as before)
        # Note: We'll update the value network separately using the buffer
        value_result = {'value_loss': 0.0}  # Placeholder for now
        
        return {
            'policy_loss_mean': np.mean(policy_losses),
            'expert_guidance_loss_mean': np.mean(expert_guidance_losses),
            'total_policy_loss_mean': np.mean(total_policy_losses),
            'entropy_loss_mean': np.mean(entropy_losses),
            'js_divergence_mean': np.mean(js_divergences),
            'beta_t_mean': np.mean(beta_values),
            'clipped_ratio_mean': np.mean(clipped_ratios),
            'value_loss': value_result['value_loss']
        }


def train_mapg_expert(env_id: str = "expert_rl/GridWorld25v0", total_episodes: int = 2000,
                     max_steps_per_ep: int = 200, buffer_size: int = 2048,
                     batch_size: int = 64, lr: float = 3e-4, gamma: float = 0.97,
                     gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                     value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                     max_grad_norm: float = 0.5, update_epochs: int = 4,
                     save_interval: int = 100, hidden_dims: List[int] = [128, 128],
                     seed: int = 1, mode: str = "mode_2", expert_guidance_coef: float = 0.05,
                     expert_beta_0: float = 0.1, expert_model_paths: List[str] = None,
                     expert_stage: str = "ep250"):
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    """Train MAPG with Expert Guidance on the GridWorld25v0 environment."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = GridWorld25v0(max_steps=max_steps_per_ep, gamma=gamma, mode=mode, seed=seed)
    obs_dim = 18  # Fixed observation dimension
    num_actions = 6  # Fixed action space size
    num_agents = 4  # Fixed number of agents
    
    # Create results directory
    results_dir = Path("mapg_expert_rl/results")
    existing_runs = list(results_dir.glob("run_*"))
    next_run_num = len(existing_runs)
    run_dir = results_dir / f"run_{next_run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    models_dir = run_dir / "models"
    configs_dir = run_dir / "configs"
    models_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Save hyperparameters including expert information
    hparams = {
        'env_id': env_id,
        'total_episodes': total_episodes,
        'max_steps_per_ep': max_steps_per_ep,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_ratio': clip_ratio,
        'value_loss_coef': value_loss_coef,
        'entropy_coef': entropy_coef,
        'max_grad_norm': max_grad_norm,
        'update_epochs': update_epochs,
        'save_interval': save_interval,
        'hidden_dims': hidden_dims,
        'seed': seed,
        'mode': mode,
        'expert_guidance_coef': expert_guidance_coef,
        'expert_beta_0': expert_beta_0,
        'expert_model_paths': expert_model_paths,
        'expert_stage': expert_stage,
        'obs_dim': obs_dim,
        'num_actions': num_actions,
        'num_agents': num_agents
    }
    
    with open(configs_dir / "hparams.json", 'w') as f:
        json.dump(hparams, f, indent=2)
    
    # Setup CSV logging with expert guidance metrics
    logs_csv = run_dir / "training_log.csv"
    with open(logs_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "episode", "steps", "return_total", "return_mean", "return_std"
        ] + [f"return_agent_{i}" for i in range(num_agents)] + [
            "policy_loss_mean", "expert_guidance_loss_mean", "total_policy_loss_mean",
            "value_loss", "entropy_loss", "js_divergence_mean", "beta_t_mean",
            "kl_div", "clipped_ratio", "buffer_size", "learning_rate"
        ]
        writer.writerow(header)
    
    # Initialize agent with expert guidance
    agent = ExpertMAPGAgent(
        obs_dim=obs_dim, num_actions=num_actions, num_agents=num_agents,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
        value_loss_coef=value_loss_coef, entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm, hidden_dims=hidden_dims, device=device,
        expert_guidance_coef=expert_guidance_coef, expert_beta_0=expert_beta_0
    )
    
    # Add better learning rate scheduling for convergence
    for i in range(num_agents):
        # Use StepLR for more stable convergence
        agent.policy_schedulers[i] = optim.lr_scheduler.StepLR(
            agent.policy_optimizers[i], step_size=500, gamma=0.8
        )
    
    # Also schedule the value optimizer
    agent.value_scheduler = optim.lr_scheduler.StepLR(
        agent.value_optimizer, step_size=500, gamma=0.8
    )
    
    # Load expert policies for the specified stage
    if expert_model_paths:
        print(f"Loading expert policies for stage {expert_stage}...")
        agent.load_expert_policies(expert_model_paths, expert_stage)
        print(f"Loaded expert policies from stage {expert_stage}")
    
    buffer = RolloutBuffer(buffer_size, obs_dim, num_agents, num_actions, device)
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    policy_losses = []
    expert_guidance_losses = []
    total_policy_losses = []
    value_losses = []
    entropy_losses = []
    js_divergences = []
    beta_values = []
    kl_divs = []
    clipped_ratios = []
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Training loop
    pbar = tqdm(range(total_episodes), desc="Training MAPG with Expert Guidance")
    best_return = float('-inf')
    
    for episode in pbar:
        obs, info = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        episode_return = 0
        step_count = 0
        
        for step in range(max_steps_per_ep):
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action
            actions_tensor, log_probs, values = agent.get_action(obs_tensor, deterministic=False)
            actions = actions_tensor.squeeze(0).cpu().numpy()
            
            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
            
            # Store in buffer (convert to tensors)
            obs_tensor = torch.FloatTensor(obs).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            rewards_tensor = torch.FloatTensor(rewards).to(device)
            values_tensor = values.squeeze(0).detach()
            log_probs_tensor = log_probs.squeeze(0).detach()
            dones_tensor = torch.BoolTensor([False]).to(device)
            
            buffer.add(obs_tensor, actions_tensor, rewards_tensor, values_tensor, log_probs_tensor, dones_tensor)
            
            # Track rewards per agent
            for i, reward in enumerate(rewards):
                episode_rewards[i].append(reward)
            
            episode_return += sum(rewards)
            obs = next_obs
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Compute discounted returns per agent
        episode_returns_agents = []
        for agent_idx in range(num_agents):
            agent_rewards = episode_rewards[agent_idx]
            agent_return = sum(agent_rewards[i] * (gamma ** i) for i in range(len(agent_rewards)))
            episode_returns_agents.append(agent_return)
        
        episode_return = sum(episode_returns_agents)
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        
        # Update buffer with final values
        if buffer.size >= buffer.capacity:
            with torch.no_grad():
                final_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _, _, final_values = agent.get_action(final_obs_tensor, deterministic=True)
                buffer.final_values = final_values.squeeze(0).detach()
        
        # Training update
        if buffer.size >= batch_size:
            # Compute GAE advantages
            with torch.no_grad():
                final_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _, _, final_values = agent.get_action(final_obs_tensor, deterministic=True)
                next_values = final_values.squeeze(0).detach()
            buffer.compute_gae(next_values, gamma, gae_lambda)
            
            # Get batch data and update with expert guidance
            update_results = []
            for batch in buffer.get_batches(batch_size):
                obs_tensor = batch['observations']
                actions_tensor = batch['actions']
                old_log_probs_tensor = batch['old_log_probs']
                advantages_tensor = batch['advantages']
                returns_tensor = batch['returns']
                
                # Update with expert guidance
                update_result = agent.update_expert_guidance(
                    obs_tensor, actions_tensor, old_log_probs_tensor,
                    advantages_tensor, returns_tensor, episode, expert_idx=0  # Use first expert
                )
                update_results.append(update_result)
            
            # Use the last update result for metrics
            update_result = update_results[-1] if update_results else {'policy_loss_mean': 0.0, 'expert_guidance_loss_mean': 0.0, 'total_policy_loss_mean': 0.0, 'value_loss': 0.0, 'entropy_loss_mean': 0.0, 'js_divergence_mean': 0.0, 'beta_t_mean': 0.0, 'clipped_ratio_mean': 0.0}
            
            # Store metrics
            policy_losses.append(update_result['policy_loss_mean'])
            expert_guidance_losses.append(update_result['expert_guidance_loss_mean'])
            total_policy_losses.append(update_result['total_policy_loss_mean'])
            value_losses.append(update_result['value_loss'])
            entropy_losses.append(update_result['entropy_loss_mean'])
            js_divergences.append(update_result['js_divergence_mean'])
            beta_values.append(update_result['beta_t_mean'])
            clipped_ratios.append(update_result['clipped_ratio_mean'])
            
            # Clear buffer
            buffer.clear()
            torch.cuda.empty_cache()
        
        # Update progress bar
        recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
        avg_return = np.mean(recent_returns)
        
        recent_policy_loss = policy_losses[-1] if policy_losses else 0.0
        recent_expert_guidance_loss = expert_guidance_losses[-1] if expert_guidance_losses else 0.0
        recent_total_policy_loss = total_policy_losses[-1] if total_policy_losses else 0.0
        recent_value_loss = value_losses[-1] if value_losses else 0.0
        recent_entropy_loss = entropy_losses[-1] if entropy_losses else 0.0
        recent_js_div = js_divergences[-1] if js_divergences else 0.0
        recent_beta = beta_values[-1] if beta_values else 0.0
        recent_clipped_ratio = clipped_ratios[-1] if clipped_ratios else 0.0
        
        current_lr = agent.policy_optimizers[0].param_groups[0]['lr']
        
        pbar.set_postfix({
            'Episode': episode,
            'Return': f'{episode_return:.2f}',
            'Avg_Return': f'{avg_return:.2f}',
            'Policy_Loss': f'{recent_policy_loss:.4f}',
            'Expert_Loss': f'{recent_expert_guidance_loss:.4f}',
            'JS_Div': f'{recent_js_div:.4f}',
            'Beta': f'{recent_beta:.4f}'
        })
        
        # Save best model
        if episode_return > best_return:
            best_return = episode_return
            # Copy shared value network to individual networks for saving
            for i in range(num_agents):
                agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
                torch.save({
                    'policy_state_dict': agent.policy_networks[i].state_dict(),
                    'value_state_dict': agent.value_networks[i].state_dict(),
                    'return': best_return
                }, models_dir / f"best_agent_{i}.pt")
        
        # Periodic checkpoint saving
        if episode % save_interval == 0:
            # Copy shared value network to individual networks for saving
            for i in range(num_agents):
                agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
                torch.save({
                    'policy_state_dict': agent.policy_networks[i].state_dict(),
                    'value_state_dict': agent.value_networks[i].state_dict(),
                    'policy_optimizer_state_dict': agent.policy_optimizers[i].state_dict()
                }, models_dir / f"checkpoint_agent_{i}_ep{episode}.pt")
        
        # Log to CSV
        with open(logs_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [
                episode, step_count, f"{episode_return:.2f}",
                f"{episode_return:.2f}", f"{0:.2f}"  # return_mean and return_std (single episode)
            ] + [f"{r:.2f}" for r in episode_returns_agents] + [
                f"{recent_policy_loss:.4f}",
                f"{recent_expert_guidance_loss:.4f}",
                f"{recent_total_policy_loss:.4f}",
                f"{recent_value_loss:.4f}",
                f"{recent_entropy_loss:.4f}",
                f"{recent_js_div:.4f}",
                f"{recent_beta:.4f}",
                f"{0:.4f}",  # kl_div (not computed)
                f"{recent_clipped_ratio:.4f}",
                buffer.size,
                f"{current_lr:.6f}"
            ]
            writer.writerow(row)
    
    # Save final models
    for i in range(num_agents):
        agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
        torch.save({
            'policy_state_dict': agent.policy_networks[i].state_dict(),
            'value_state_dict': agent.value_networks[i].state_dict(),
            'policy_optimizer_state_dict': agent.policy_optimizers[i].state_dict()
        }, models_dir / f"final_agent_{i}.pt")
    
    # Create training plots with expert guidance metrics
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode returns
        axes[0, 0].plot(episode_returns, alpha=0.3, color='blue')
        if len(episode_returns) > 20:
            window = min(20, len(episode_returns) // 10)
            moving_avg = pd.Series(episode_returns).rolling(window=window, center=True).mean()
            axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window})')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Policy loss
        if policy_losses:
            axes[0, 1].plot(policy_losses, alpha=0.7, color='green', label='Policy Loss')
            axes[0, 1].plot(expert_guidance_losses, alpha=0.7, color='orange', label='Expert Guidance Loss')
            axes[0, 1].set_title('Policy Losses')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Jensen-Shannon Divergence
        if js_divergences:
            axes[0, 2].plot(js_divergences, alpha=0.7, color='purple')
            axes[0, 2].set_title('Jensen-Shannon Divergence')
            axes[0, 2].set_xlabel('Update Step')
            axes[0, 2].set_ylabel('JS Divergence')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Beta values
        if beta_values:
            axes[1, 0].plot(beta_values, alpha=0.7, color='red')
            axes[1, 0].set_title('Beta_t (Expert Guidance Weight)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Beta_t')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Value loss
        if value_losses:
            axes[1, 1].plot(value_losses, alpha=0.7, color='cyan')
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Clipped ratio
        if clipped_ratios:
            axes[1, 2].plot(clipped_ratios, alpha=0.7, color='brown')
            axes[1, 2].set_title('Clipped Ratio')
            axes[1, 2].set_xlabel('Update Step')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(run_dir / "training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error saving plots: {e}")
        plt.close()
    
    # Clear GPU memory before final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Calculate training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Create training log directory
    log_dir = run_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    # Save comprehensive training log with expert information
    training_log = {
        "total_training_time_seconds": float(total_training_time),
        "total_training_time_hours": float(total_training_time / 3600),
        "total_episodes": int(total_episodes),
        "best_return": float(best_return),
        "expert_guidance_info": {
            "expert_stage": expert_stage,
            "expert_guidance_coef": float(expert_guidance_coef),
            "expert_beta_0": float(expert_beta_0),
            "expert_model_paths": expert_model_paths,
            "final_js_divergence_mean": float(np.mean(js_divergences[-10:])) if js_divergences else 0.0,
            "final_beta_t": float(beta_values[-1]) if beta_values else 0.0,
            "expert_guidance_effectiveness": {
                "avg_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "avg_expert_guidance_loss": float(np.mean(expert_guidance_losses)) if expert_guidance_losses else 0.0,
                "avg_js_divergence": float(np.mean(js_divergences)) if js_divergences else 0.0
            }
        }
    }
    
    try:
        log_path = log_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
    except Exception as e:
        print(f"Error saving training log: {e}")
    
    env.close()
    print(f"\nTraining completed! Results saved to: {run_dir}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"Final average return (last 100 episodes): {np.mean(episode_returns[-100:]):.2f}")
    print(f"Final JS Divergence: {np.mean(js_divergences[-10:]) if js_divergences else 0.0:.4f}")
    print(f"Final Beta_t: {beta_values[-1] if beta_values else 0.0:.4f}")
    
    # Final cleanup
    try:
        # Clear all variables if they exist
        if 'buffer' in locals():
            del buffer
        if 'env' in locals():
            del env
        gc.collect()
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return agent, episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPG with Expert Guidance on GridWorld25v0")
    parser.add_argument("--total_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--buffer_size", type=int, default=2048, help="Rollout buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update_epochs", type=int, default=4, help="Update epochs per batch")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--mode", type=str, default="mode_2", help="Environment mode")
    parser.add_argument("--expert_guidance_coef", type=float, default=0.05, help="Expert guidance coefficient")
    parser.add_argument("--expert_beta_0", type=float, default=0.1, help="Expert beta_0 parameter")
    parser.add_argument("--expert_model_paths", nargs='+', 
                       default=["madqn/results/run_25/models/checkpoint_agent_0_ep250.pt",
                               "madqn/results/run_25/models/checkpoint_agent_0_ep500.pt",
                               "madqn/results/run_25/models/checkpoint_agent_0_ep1500.pt"],
                       help="Paths to expert model directories")
    parser.add_argument("--expert_stage", type=str, default="all",
                       choices=["ep250", "ep500", "ep1500", "all"],
                       help="Which expert stage to use: ep250, ep500, ep1500, or all (run all three)")
    
    args = parser.parse_args()
    
    # Handle multiple expert stages if "all" is selected
    if args.expert_stage == "all":
        expert_stages = ["ep250", "ep500", "ep1500"]
        print(f"Running training for all expert stages: {expert_stages}")
        
        # Track total time for all runs
        total_start_time = time.time()
        stage_results = []
        
        for i, stage in enumerate(expert_stages, 1):
            print(f"\n{'='*60}")
            print(f"Starting training with expert stage: {stage} ({i}/{len(expert_stages)})")
            print(f"{'='*60}")
            
            stage_start_time = time.time()
            agent, returns = train_mapg_expert(
                total_episodes=args.total_episodes,
                max_steps_per_ep=args.max_steps_per_ep,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                lr=args.lr,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_ratio=args.clip_ratio,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                update_epochs=args.update_epochs,
                save_interval=args.save_interval,
                seed=args.seed,
                mode=args.mode,
                expert_guidance_coef=args.expert_guidance_coef,
                expert_beta_0=args.expert_beta_0,
                expert_model_paths=args.expert_model_paths,
                expert_stage=stage
            )
            
            stage_end_time = time.time()
            stage_time = stage_end_time - stage_start_time
            final_return = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
            
            stage_results.append({
                'stage': stage,
                'final_return': final_return,
                'training_time_seconds': stage_time,
                'training_time_hours': stage_time / 3600
            })
            
            print(f"\nCompleted training with expert stage: {stage}")
            print(f"Final average return: {final_return:.2f}")
            print(f"Training time: {stage_time/3600:.2f} hours ({stage_time:.1f} seconds)")
        
        # Calculate and display total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL EXPERT STAGES")
        print(f"{'='*60}")
        print(f"Total time for all stages: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
        print(f"Average time per stage: {(total_time/len(expert_stages))/3600:.2f} hours ({(total_time/len(expert_stages)):.1f} seconds)")
        print("\nIndividual stage results:")
        for result in stage_results:
            print(f"  {result['stage']}: Return={result['final_return']:.2f}, Time={result['training_time_hours']:.2f}h")
        
        print(f"\n✓ All expert stage training completed successfully!")
    else:
        # Single expert stage training
        agent, returns = train_mapg_expert(
            total_episodes=args.total_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            save_interval=args.save_interval,
            seed=args.seed,
            mode=args.mode,
            expert_guidance_coef=args.expert_guidance_coef,
            expert_beta_0=args.expert_beta_0,
            expert_model_paths=args.expert_model_paths,
            expert_stage=args.expert_stage
        )
