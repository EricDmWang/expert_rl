#!/usr/bin/env python3
"""
MAPG Training with LBF Expert Guidance
=====================================

This script trains MAPG agents using the lbf_small.pt model as expert guidance.
The expert model is applied to all agents with Jensen-Shannon divergence regularization.
"""

import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from expert_rl.gridworld25_env import GridWorld25v0
from lbf_llm.lbf_wrapper import LBFCustomObsWrapper
from madqn.train_madqn import ImprovedDQNNetwork
from mapg_expert_rl.train_mapg import MAPGAgent, RolloutBuffer, PolicyNetwork, ValueNetwork

# Add lbf_llm to path for SmallTabNet import
sys.path.append('/home/dongmingwang/project/Expert_RL/lbf_llm')

# Import SmallTabNet directly by copying the class definition to avoid relative import issues
class SmallTabNet(nn.Module):
    def __init__(self, in_dim=18, hidden1=64, hidden2=64, out_dim=6, p_drop=0.1, noise=0.05):
        super().__init__()
        self.noise = GaussianNoise(noise)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, out_dim),
        )
    def forward(self, x):
        # ensure input is on the same device as model parameters
        param_device = next(self.parameters()).device
        if x.device != param_device:
            x = x.to(param_device)
        x = self.noise(x)
        return self.mlp(x)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.05):
        super().__init__(); self.sigma = float(sigma)
    def forward(self, x):
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


class LBFExpertMAPGAgent(MAPGAgent):
    """MAPG Agent with LBF expert guidance using Jensen-Shannon divergence."""
    
    def __init__(self, obs_dim: int, num_actions: int, num_agents: int, 
                 lr: float = 3e-4, gamma: float = 0.97, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 hidden_dims: List[int] = [128, 128], device: torch.device = None,
                 expert_guidance_coef: float = 0.05, expert_beta_0: float = 0.1):
        super().__init__(obs_dim, num_actions, num_agents, lr, gamma, gae_lambda,
                        clip_ratio, value_loss_coef, entropy_coef, max_grad_norm,
                        hidden_dims, device)
        
        # Expert guidance parameters
        self.expert_guidance_coef = expert_guidance_coef
        self.expert_beta_0 = expert_beta_0
        self.expert_policies = None  # Will be loaded later
        self.expert_mu = None
        self.expert_std = None
        
    def load_lbf_expert_policy(self, lbf_model_path: str):
        """Load the lbf_small.pt model as expert policy for all agents."""
        try:
            # Load the LBF model checkpoint
            checkpoint = torch.load(lbf_model_path, map_location=self.device)
            
            # Extract normalization parameters
            self.expert_mu = checkpoint['mu'].to(self.device)
            self.expert_std = checkpoint['std'].to(self.device)
            
            # Create expert networks for all agents using the same LBF model
            self.expert_policies = []
            for agent_idx in range(self.num_agents):
                agent_experts = []
                
                # Create SmallTabNet with the same architecture as the saved model
                expert_net = SmallTabNet(
                    in_dim=self.obs_dim,
                    hidden1=64,
                    hidden2=64,
                    out_dim=self.num_actions,
                    p_drop=0.0,  # No dropout during inference
                    noise=0.0    # No noise during inference
                ).to(self.device)
                
                # Load the state dict
                expert_net.load_state_dict(checkpoint['state_dict'])
                expert_net.eval()  # Set to evaluation mode
                
                agent_experts.append(expert_net)
                print(f"Loaded LBF expert policy for agent {agent_idx} from {lbf_model_path}")
                
                self.expert_policies.append(agent_experts)
            
            print(f"Loaded LBF expert policy for all {self.num_agents} agents from {lbf_model_path}")
            
        except Exception as e:
            print(f"Error loading LBF expert policy: {e}")
            # Fallback to no expert networks
            self.expert_policies = [[None] for _ in range(self.num_agents)]
    
    def compute_jensen_shannon_divergence(self, policy_dist: torch.distributions.Categorical, 
                                        expert_dist: torch.distributions.Categorical) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between current and expert policies."""
        # Get probability distributions (clone to avoid inplace operations)
        p = policy_dist.probs.clone()
        q = expert_dist.probs.clone()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p = p + eps
        q = q + eps
        
        # Normalize to ensure they sum to 1 (avoid inplace division)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        
        # Jensen-Shannon divergence: 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(p.log(), m, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(q.log(), m, reduction='none').sum(dim=-1)
        
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div
    
    def get_expert_action_distribution(self, obs: torch.Tensor, agent_idx: int, expert_idx: int = 0) -> torch.distributions.Categorical:
        """Get action distribution from LBF expert policy."""
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
            # Apply LBF normalization
            obs_normalized = (obs - self.expert_mu) / self.expert_std
            expert_output = expert_net(obs_normalized)
            
            # Convert to action probabilities using softmax with temperature scaling
            # Higher temperature = more exploration, lower temperature = more exploitation
            temperature = 2.0  # Increased temperature for smoother expert guidance
            expert_logits = expert_output / temperature
            expert_dist = torch.distributions.Categorical(logits=expert_logits)
        
        return expert_dist
    
    def _update_policy_agent_expert(self, agent_idx: int, obs: torch.Tensor, actions: torch.Tensor,
                                  old_log_probs: torch.Tensor, advantages: torch.Tensor,
                                  returns: torch.Tensor, episode: int, expert_idx: int = 0):
        """Update policy for a specific agent with LBF expert guidance."""
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
        
        # Total policy loss (remove expert_guidance_coef to avoid double scaling)
        total_policy_loss = policy_loss + expert_guidance_loss
        
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
    
    def _update_value_network_batch(self, obs: torch.Tensor, returns: torch.Tensor):
        """Update value network with a batch of data for better stability."""
        value_net = self.shared_value_network
        optimizer = self.value_optimizer
        scheduler = self.value_scheduler
        
        # Get current value estimates
        values = value_net(obs.view(-1, self.obs_dim)).view(obs.shape[0], self.num_agents)
        
        # Compute value loss (MSE between predicted and actual returns)
        value_loss = F.mse_loss(values, returns)
        
        # Update
        optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        return {'value_loss': value_loss.item()}


def train_mapg_llm(total_episodes: int = 2000, max_steps_per_ep: int = 200,
                   buffer_size: int = 4096, batch_size: int = 128, lr: float = 3e-4,
                   gamma: float = 0.97, gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                   value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                   max_grad_norm: float = 0.5, update_epochs: int = 4,
                   save_interval: int = 100, seed: int = 1, mode: str = "mode_2",
                   expert_guidance_coef: float = 0.5, expert_beta_0: float = 0.1,
                   lbf_model_path: str = "lbf_llm/lbf_small.pt") -> Tuple[LBFExpertMAPGAgent, List[float]]:
    """Train MAPG agents with LBF expert guidance."""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld25v0(mode=mode)
    # Don't use LBFCustomObsWrapper for GridWorld25v0 as it's not compatible
    # The GridWorld25v0 already provides the correct observation format
    obs, _ = env.reset(seed=seed)
    
    print(f"Environment: {env.num_agents} agents, obs_dim={obs.shape[-1]}, actions={env.action_space.n}")
    
    # Create results directory
    results_dir = Path("mapg_expert_rl/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next run number
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_numbers = [int(d.name.split("_")[1]) for d in run_dirs if d.name.split("_")[1].isdigit()]
    next_run = max(run_numbers, default=-1) + 1
    run_dir = results_dir / f"run_{next_run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Create agent
    agent = LBFExpertMAPGAgent(
        obs_dim=obs.shape[-1],
        num_actions=env.action_space.n,
        num_agents=env.num_agents,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        hidden_dims=[128, 128],
        device=device,
        expert_guidance_coef=expert_guidance_coef,
        expert_beta_0=expert_beta_0
    )
    
    # Load LBF expert policy
    agent.load_lbf_expert_policy(lbf_model_path)
    
    # Add better learning rate scheduling for convergence (match expert implementation)
    for i in range(env.num_agents):
        # Use StepLR for more stable convergence
        agent.policy_schedulers[i] = optim.lr_scheduler.StepLR(
            agent.policy_optimizers[i], step_size=500, gamma=0.9
        )
    
    # Also schedule the value optimizer
    agent.value_scheduler = optim.lr_scheduler.StepLR(
        agent.value_optimizer, step_size=500, gamma=0.9
    )
    
    # Create rollout buffer (capacity, obs_dim, num_agents, num_actions, device)
    buffer = RolloutBuffer(buffer_size, obs.shape[-1], env.num_agents, env.action_space.n, device)
    
    # Training metrics
    episode_returns = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    expert_guidance_losses = []
    js_divergences = []
    beta_values = []
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Training loop
    pbar = tqdm(range(total_episodes), desc="Training MAPG with LBF Expert")
    best_return = float('-inf')
    
    for episode in pbar:
        # Collect rollout
        obs, _ = env.reset(seed=seed + episode)
        episode_rewards = [[] for _ in range(env.num_agents)]
        episode_return = 0.0

        for step in range(max_steps_per_ep):
            # Convert obs to tensor with batch dimension
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get actions from agent (CTDE style)
            actions_tensor, log_probs, values = agent.get_action(obs_tensor, deterministic=False)
            actions = actions_tensor.squeeze(0).cpu().numpy().tolist()
            
            # Step environment
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # Add to buffer (store current obs row)
            buffer.add(
                obs=obs_tensor.squeeze(0),
                actions=actions_tensor.squeeze(0),
                rewards=torch.FloatTensor(rewards).to(device),
                values=values.squeeze(0),
                log_probs=log_probs.squeeze(0),
                dones=torch.tensor(done, device=device)
            )
            
            # track raw rewards per agent for discounted episodic return later
            for i in range(env.num_agents):
                episode_rewards[i].append(rewards[i])
            
            obs = next_obs
            if done:
                break
        
        # Calculate discounted episodic return per agent, then sum (consistent with expert trainer)
        episode_returns_agents = []
        for i in range(env.num_agents):
            agent_return = sum((gamma ** t) * r for t, r in enumerate(episode_rewards[i]))
            episode_returns_agents.append(agent_return)
        episode_return = float(sum(episode_returns_agents))
        episode_returns.append(episode_return)
        best_return = max(best_return, episode_return)
        
        # Update progress bar
        avg_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else episode_return
        pbar.set_postfix({
            'Episode': episode,
            'Return': f'{episode_return:.2f}',
            'Avg_Return': f'{avg_return:.2f}',
            'Best_Return': f'{best_return:.2f}'
        })
        
        # Training update (match expert implementation)
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
                
                # Update with expert guidance per agent
                total_policy_loss = 0.0
                total_entropy_loss = 0.0
                total_js_div = 0.0
                total_beta_t = 0.0
                
                for agent_idx in range(env.num_agents):
                    # Clone tensors to avoid inplace operations on buffer data
                    agent_obs = obs_tensor[:, agent_idx].clone().detach()
                    agent_actions = actions_tensor[:, agent_idx].clone().detach()
                    agent_old_log_probs = old_log_probs_tensor[:, agent_idx].clone().detach()
                    agent_advantages = advantages_tensor[:, agent_idx].clone().detach()
                    agent_returns = returns_tensor[:, agent_idx].clone().detach()
                    
                    # Update individual agent with expert guidance
                    metrics = agent._update_policy_agent_expert(
                        agent_idx, agent_obs, agent_actions, agent_old_log_probs,
                        agent_advantages, agent_returns, episode, expert_idx=0
                    )
                    total_policy_loss += metrics['policy_loss']
                    total_entropy_loss += metrics['entropy_loss']
                    total_js_div += metrics['js_divergence']
                    total_beta_t += metrics['beta_t']
                
                # Update shared value network (clone to avoid gradient sharing)
                value_metrics = agent._update_value_network_batch(
                    obs_tensor.clone().detach(), returns_tensor.clone().detach()
                )
                
                update_results.append({
                    'policy_loss_mean': total_policy_loss / env.num_agents,
                    'entropy_loss_mean': total_entropy_loss / env.num_agents,
                    'js_divergence_mean': total_js_div / env.num_agents,
                    'beta_t_mean': total_beta_t / env.num_agents,
                    'value_loss': value_metrics['value_loss']
                })
            
            # Use the last update result for metrics
            update_result = update_results[-1] if update_results else {
                'policy_loss_mean': 0.0, 'entropy_loss_mean': 0.0, 'js_divergence_mean': 0.0, 
                'beta_t_mean': 0.0, 'value_loss': 0.0
            }
            
            # Store metrics
            policy_losses.append(update_result['policy_loss_mean'])
            entropy_losses.append(update_result['entropy_loss_mean'])
            js_divergences.append(update_result['js_divergence_mean'])
            beta_values.append(update_result['beta_t_mean'])
            value_losses.append(update_result['value_loss'])
            
            # Clear buffer
            buffer.clear()
            torch.cuda.empty_cache()
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            models_dir = run_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Copy shared value to individual value networks before saving
            for i in range(agent.num_agents):
                agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())

            # Save in expert-style single-file per agent
            for agent_idx in range(env.num_agents):
                checkpoint = {
                    'episode': int(episode),
                    'policy_state_dict': agent.policy_networks[agent_idx].state_dict(),
                    'value_state_dict': agent.value_networks[agent_idx].state_dict(),
                    'policy_optimizer_state_dict': agent.policy_optimizers[agent_idx].state_dict(),
                    'return': float(episode_return)
                }
                torch.save(checkpoint, models_dir / f"checkpoint_agent_{agent_idx}_ep{episode}.pt")
    
    # Save final models
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Copy shared value to individual value networks before final save
    for i in range(agent.num_agents):
        agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())

    # Save in expert-style single-file per agent
    for agent_idx in range(env.num_agents):
        final_ckpt = {
            'episode': int(total_episodes),
            'policy_state_dict': agent.policy_networks[agent_idx].state_dict(),
            'value_state_dict': agent.value_networks[agent_idx].state_dict(),
            'policy_optimizer_state_dict': agent.policy_optimizers[agent_idx].state_dict(),
            'return': float(episode_return)
        }
        torch.save(final_ckpt, models_dir / f"final_agent_{agent_idx}.pt")
    
    # Save hyperparameters
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    hparams = {
        "total_episodes": int(total_episodes),
        "max_steps_per_ep": int(max_steps_per_ep),
        "buffer_size": int(buffer_size),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_ratio": float(clip_ratio),
        "value_loss_coef": float(value_loss_coef),
        "entropy_coef": float(entropy_coef),
        "max_grad_norm": float(max_grad_norm),
        "update_epochs": int(update_epochs),
        "save_interval": int(save_interval),
        "seed": int(seed),
        "mode": str(mode),
        "expert_guidance_coef": float(expert_guidance_coef),
        "expert_beta_0": float(expert_beta_0),
        "lbf_model_path": str(lbf_model_path),
        "obs_dim": int(obs.shape[-1]),
        "num_actions": int(env.action_space.n),
        "num_agents": int(env.num_agents),
        "hidden_dims": [128, 128],
        "device": str(device)
    }
    
    with open(configs_dir / "hparams.json", 'w') as f:
        json.dump(hparams, f, indent=2)
    
    # Save training progress to CSV (match expert implementation format)
    csv_path = run_dir / "training_log.csv"
    with open(csv_path, 'w') as f:
        header = [
            "episode", "steps", "return_total", "return_mean", "return_std"
        ] + [f"return_agent_{i}" for i in range(env.num_agents)] + [
            "policy_loss_mean", "expert_guidance_loss_mean", "total_policy_loss_mean",
            "value_loss", "entropy_loss", "js_divergence_mean", "beta_t_mean",
            "kl_div", "clipped_ratio", "buffer_size", "learning_rate"
        ]
        f.write(",".join(header) + "\n")
        
        # Write data rows
        for i in range(len(episode_returns)):
            row = [
                str(i),  # episode
                str(max_steps_per_ep),  # steps (approximate)
                f"{episode_returns[i]:.2f}",  # return_total
                f"{episode_returns[i]:.2f}",  # return_mean (same as total for single episode)
                f"{0:.2f}"  # return_std (0 for single episode)
            ] + [f"{episode_returns[i]/env.num_agents:.2f}" for _ in range(env.num_agents)] + [  # return_agent_X
                f"{policy_losses[i]:.4f}" if i < len(policy_losses) else "0.0000",  # policy_loss_mean
                f"{0:.4f}",  # expert_guidance_loss_mean (not tracked separately)
                f"{policy_losses[i]:.4f}" if i < len(policy_losses) else "0.0000",  # total_policy_loss_mean
                f"{value_losses[i]:.4f}" if i < len(value_losses) else "0.0000",  # value_loss
                f"{entropy_losses[i]:.4f}" if i < len(entropy_losses) else "0.0000",  # entropy_loss
                f"{js_divergences[i]:.4f}" if i < len(js_divergences) else "0.0000",  # js_divergence_mean
                f"{beta_values[i]:.4f}" if i < len(beta_values) else "0.0000",  # beta_t_mean
                f"{0:.4f}",  # kl_div (not computed)
                f"{1.0:.4f}",  # clipped_ratio (approximate)
                str(buffer_size),  # buffer_size
                f"{lr:.6f}"  # learning_rate
            ]
            f.write(",".join(row) + "\n")
    
    # Create training plots
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MAPG Training Progress with LBF Expert Guidance', fontsize=16)
        
        # Episode returns
        axes[0, 0].plot(episode_returns)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Policy loss
        if policy_losses:
            axes[0, 1].plot(policy_losses)
            axes[0, 1].set_title('Policy Loss')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Value loss
        if value_losses:
            axes[0, 2].plot(value_losses)
            axes[0, 2].set_title('Value Loss')
            axes[0, 2].set_xlabel('Update Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Expert guidance loss
        if expert_guidance_losses:
            axes[1, 0].plot(expert_guidance_losses)
            axes[1, 0].set_title('Expert Guidance Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # JS Divergence
        if js_divergences:
            axes[1, 1].plot(js_divergences)
            axes[1, 1].set_title('Jensen-Shannon Divergence')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('JS Divergence')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Beta_t values
        if beta_values:
            axes[1, 2].plot(beta_values)
            axes[1, 2].set_title('Beta_t (Expert Guidance Weight)')
            axes[1, 2].set_xlabel('Update Step')
            axes[1, 2].set_ylabel('Beta_t')
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
    
    # Save comprehensive training log with LBF expert information
    training_log = {
        "total_training_time_seconds": float(total_training_time),
        "total_training_time_hours": float(total_training_time / 3600),
        "total_episodes": int(total_episodes),
        "best_return": float(best_return),
        "lbf_expert_info": {
            "lbf_model_path": lbf_model_path,
            "expert_guidance_coef": float(expert_guidance_coef),
            "expert_beta_0": float(expert_beta_0),
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
        print(f"Training log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving training log: {e}")
    
    # Cleanup (preserve agent before cleanup for return)
    return_agent = agent
    env.close()
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    
    print(f"\nTraining completed! Results saved to: {run_dir}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"Final average return (last 100 episodes): {np.mean(episode_returns[-100:]):.2f}")
    if js_divergences:
        print(f"Final JS Divergence: {js_divergences[-1]:.4f}")
    if beta_values:
        print(f"Final Beta_t: {beta_values[-1]:.4f}")
    
    return return_agent, episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPG with LBF Expert Guidance")
    
    # Training parameters
    parser.add_argument("--total_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--buffer_size", type=int, default=4096, help="Rollout buffer size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--update_epochs", type=int, default=4, help="Update epochs per rollout")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    
    # Environment parameters
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--mode", type=str, default="mode_2", help="Environment mode")
    
    # Expert guidance parameters
    parser.add_argument("--expert_guidance_coef", type=float, default=0.05, help="Expert guidance coefficient")
    parser.add_argument("--expert_beta_0", type=float, default=0.1, help="Initial expert guidance weight")
    parser.add_argument("--lbf_model_path", type=str, default="lbf_llm/lbf_small.pt", help="Path to LBF expert model")
    
    args = parser.parse_args()
    
    # Train MAPG with LBF expert
    agent, returns = train_mapg_llm(
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
        lbf_model_path=args.lbf_model_path
    )
    
    print("Cleanup completed successfully.")
