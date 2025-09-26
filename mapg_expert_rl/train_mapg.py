#!/usr/bin/env python3
"""
MAPG Training Script for Expert RL GridWorld25v0 Environment

Multi-Agent Policy Gradient (MAPG) implementation with stability improvements
for cooperative multi-agent reinforcement learning.
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

from expert_rl import GridWorld25v0


class RolloutBuffer:
    """Rollout buffer for MAPG experience storage."""
    
    def __init__(self, capacity: int, obs_dim: int, num_agents: int, num_actions: int, 
                 device: torch.device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.device = device
        
        # Storage tensors
        self.observations = torch.zeros((capacity, num_agents, obs_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((capacity, num_agents), device=device, dtype=torch.long)
        self.rewards = torch.zeros((capacity, num_agents), device=device, dtype=torch.float32)
        self.values = torch.zeros((capacity, num_agents), device=device, dtype=torch.float32)
        self.log_probs = torch.zeros((capacity, num_agents), device=device, dtype=torch.float32)
        self.dones = torch.zeros((capacity,), device=device, dtype=torch.bool)
        self.advantages = torch.zeros((capacity, num_agents), device=device, dtype=torch.float32)
        self.returns = torch.zeros((capacity, num_agents), device=device, dtype=torch.float32)
        
        self.size = 0
        self.ptr = 0
    
    def add(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
            values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor):
        """Add experience to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_gae(self, next_values: torch.Tensor, gamma: float = 0.95, 
                    gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        last_advantage = torch.zeros_like(next_values)
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        returns = advantages + self.values
        
        # Normalize advantages per agent
        for i in range(self.num_agents):
            agent_advantages = advantages[:, i]
            if agent_advantages.std() > 1e-8:
                advantages[:, i] = (agent_advantages - agent_advantages.mean()) / agent_advantages.std()
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int):
        """Generate batches for training."""
        indices = torch.randperm(self.size, device=self.device)
        for start_idx in range(0, self.size, batch_size):
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'old_log_probs': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }
    
    def clear(self):
        """Clear buffer."""
        self.size = 0
        self.ptr = 0


class PolicyNetwork(nn.Module):
    """Policy network (actor) for MAPG CTDE."""
    
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # Build network layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Policy head only
        self.policy_head = nn.Linear(prev_dim, num_actions)
        
        # Shared layers
        self.shared_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)  # Conservative initialization
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        shared_features = self.shared_layers(x)
        
        # Policy logits
        policy_logits = self.policy_head(shared_features)
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        
        return policy_dist


class ValueNetwork(nn.Module):
    """Value network (critic) for MAPG CTDE."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        
        # Build network layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Shared layers
        self.shared_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)  # Conservative initialization
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        shared_features = self.shared_layers(x)
        
        # Value
        value = self.value_head(shared_features).squeeze(-1)
        
        return value


class MAPGAgent:
    """MAPG agent with CTDE (Centralized Training, Decentralized Execution) architecture."""
    
    def __init__(self, obs_dim: int, num_actions: int, num_agents: int,
                 lr: float = 3e-4, gamma: float = 0.95, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 hidden_dims: List[int] = [128, 128], device: torch.device = None):
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Individual policy networks (actors) for each agent
        self.policy_networks = nn.ModuleList([
            PolicyNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Shared value network (critic) for centralized training
        self.shared_value_network = ValueNetwork(obs_dim, hidden_dims).to(self.device)
        
        # Individual value networks for saving (copies of shared network)
        self.value_networks = nn.ModuleList([
            ValueNetwork(obs_dim, hidden_dims).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Optimizers for policy networks
        self.policy_optimizers = [
            optim.Adam(network.parameters(), lr=lr, eps=1e-5)
            for network in self.policy_networks
        ]
        
        # Optimizer for shared value network
        self.value_optimizer = optim.Adam(self.shared_value_network.parameters(), lr=lr, eps=1e-5)
        
        # Learning rate schedulers
        self.policy_schedulers = [
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-6
            )
            for optimizer in self.policy_optimizers
        ]
        self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode='min', factor=0.8, patience=20, min_lr=1e-6
        )
        
        self.update_count = 0
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get action from policy using CTDE architecture."""
        batch_size = obs.size(0)
        actions = torch.zeros((batch_size, self.num_agents), dtype=torch.long, device=self.device)
        log_probs = torch.zeros((batch_size, self.num_agents), device=self.device)
        values = torch.zeros((batch_size, self.num_agents), device=self.device)
        
        # Get actions from individual policy networks
        for i in range(self.num_agents):
            policy_dist = self.policy_networks[i](obs[:, i])
            
            if deterministic:
                action = policy_dist.probs.argmax(dim=-1)
            else:
                action = policy_dist.sample()
            
            log_prob = policy_dist.log_prob(action)
            
            actions[:, i] = action
            log_probs[:, i] = log_prob
        
        # Get values from shared value network
        for i in range(self.num_agents):
            values[:, i] = self.shared_value_network(obs[:, i])
        
        return actions, log_probs, values
    
    def evaluate_action(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for given observations using CTDE architecture."""
        batch_size = obs.size(0)
        log_probs = torch.zeros((batch_size, self.num_agents), device=self.device)
        values = torch.zeros((batch_size, self.num_agents), device=self.device)
        entropies = torch.zeros((batch_size, self.num_agents), device=self.device)
        
        # Get log probs and entropy from individual policy networks
        for i in range(self.num_agents):
            policy_dist = self.policy_networks[i](obs[:, i])
            log_prob = policy_dist.log_prob(actions[:, i])
            entropy = policy_dist.entropy()
            
            log_probs[:, i] = log_prob
            entropies[:, i] = entropy
        
        # Get values from shared value network
        for i in range(self.num_agents):
            values[:, i] = self.shared_value_network(obs[:, i])
        
        return log_probs, values, entropies
    
    def update(self, buffer: RolloutBuffer, epochs: int = 4):
        """Update policy and value networks using CTDE PPO."""
        if buffer.size == 0:
            return {}
        
        # Compute advantages and returns
        with torch.no_grad():
            # Get next value estimates (zero for terminal states)
            next_values = torch.zeros((self.num_agents,), device=self.device)
            buffer.compute_gae(next_values, self.gamma, self.gae_lambda)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_clipped_ratio = 0.0
        
        # Update policy networks (individual for each agent)
        for agent_idx in range(self.num_agents):
            policy_losses = self._update_policy_agent(agent_idx, buffer, epochs)
            
            total_policy_loss += policy_losses['policy_loss']
            total_entropy_loss += policy_losses['entropy_loss']
            total_kl_div += policy_losses['kl_div']
            total_clipped_ratio += policy_losses['clipped_ratio']
        
        # Update shared value network
        value_losses = self._update_value_network(buffer, epochs)
        total_value_loss = value_losses['value_loss']
        
        self.update_count += 1
        
        # Average losses across agents
        avg_losses = {
            'policy_loss': total_policy_loss / self.num_agents,
            'value_loss': total_value_loss,
            'entropy_loss': total_entropy_loss / self.num_agents,
            'kl_div': total_kl_div / self.num_agents,
            'clipped_ratio': total_clipped_ratio / self.num_agents
        }
        
        return avg_losses
    
    def _update_policy_agent(self, agent_idx: int, buffer: RolloutBuffer, epochs: int):
        """Update individual agent policy network."""
        network = self.policy_networks[agent_idx]
        optimizer = self.policy_optimizers[agent_idx]
        
        # Get data for this agent
        obs = buffer.observations[:buffer.size, agent_idx].detach()
        actions = buffer.actions[:buffer.size, agent_idx].detach()
        old_log_probs = buffer.log_probs[:buffer.size, agent_idx].detach()
        advantages = buffer.advantages[:buffer.size, agent_idx].detach()
        
        # Normalize advantages for this agent
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_clipped_ratio = 0.0
        
        for epoch in range(epochs):
            # Get current policy predictions
            policy_dist = network(obs)
            new_log_probs = policy_dist.log_prob(actions)
            entropy = policy_dist.entropy()
            
            # Compute policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy loss (encourage exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss (only policy and entropy)
            loss = policy_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            # Compute statistics
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean()
                clipped_ratio = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_div += kl_div.item()
                total_clipped_ratio += clipped_ratio.item()
        
        # Update learning rate
        self.policy_schedulers[agent_idx].step(total_policy_loss / epochs)
        
        return {
            'policy_loss': total_policy_loss / epochs,
            'entropy_loss': total_entropy_loss / epochs,
            'kl_div': total_kl_div / epochs,
            'clipped_ratio': total_clipped_ratio / epochs
        }
    
    def _update_value_network(self, buffer: RolloutBuffer, epochs: int):
        """Update shared value network."""
        optimizer = self.value_optimizer
        
        # Get data for all agents
        obs = buffer.observations[:buffer.size].detach()
        returns = buffer.returns[:buffer.size].detach()
        
        total_value_loss = 0.0
        
        for epoch in range(epochs):
            # Compute value loss for all agents
            epoch_value_loss = 0.0
            
            for agent_idx in range(self.num_agents):
                agent_obs = obs[:, agent_idx]
                agent_returns = returns[:, agent_idx]
                
                # Get current value estimates
                values = self.shared_value_network(agent_obs)
                
                # Value loss
                value_loss = F.mse_loss(values, agent_returns)
                epoch_value_loss += value_loss
            
            # Average value loss across agents
            epoch_value_loss = epoch_value_loss / self.num_agents
            
            # Optimize
            optimizer.zero_grad(set_to_none=True)
            epoch_value_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.shared_value_network.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            total_value_loss += epoch_value_loss.item()
        
        # Update learning rate
        self.value_scheduler.step(total_value_loss / epochs)
        
        return {
            'value_loss': total_value_loss / epochs
        }


def train_mapg(env_id: str = "expert_rl/GridWorld25v0", total_episodes: int = 1000,
               max_steps_per_ep: int = 200, buffer_size: int = 2048,
               batch_size: int = 64, lr: float = 3e-4, gamma: float = 0.95,
               gae_lambda: float = 0.95, clip_ratio: float = 0.2,
               value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
               max_grad_norm: float = 0.5, update_epochs: int = 4,
               save_interval: int = 100, hidden_dims: List[int] = [128, 128],
               seed: int = 1, mode: str = "mode_2"):
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    """Train MAPG on the GridWorld25v0 environment."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld25v0(max_steps=max_steps_per_ep, gamma=gamma)
    obs, info = env.reset()
    
    obs_dim = obs.shape[1]  # 18
    num_actions = 6
    num_agents = 4
    
    # Create results directory
    results_dir = Path("mapg_expert_rl/results")
    run_idx = len(list(results_dir.glob("run_*"))) if results_dir.exists() else 0
    run_dir = results_dir / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Save hyperparameters
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
        'hidden_dims': hidden_dims,
        'seed': seed,
        'mode': mode,
        'obs_dim': obs_dim,
        'num_actions': num_actions,
        'num_agents': num_agents
    }
    
    with open(configs_dir / "hparams.json", 'w') as f:
        json.dump(hparams, f, indent=2)
    
    # Setup CSV logging
    logs_csv = run_dir / "training_log.csv"
    with open(logs_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "episode", "steps", "return_total", "return_mean", "return_std"
        ] + [f"return_agent_{i}" for i in range(num_agents)] + [
            "policy_loss_mean", "value_loss", "entropy_loss", "kl_div", "clipped_ratio",
            "buffer_size", "learning_rate"
        ]
        writer.writerow(header)
    
    # Initialize agent and buffer
    agent = MAPGAgent(
        obs_dim=obs_dim, num_actions=num_actions, num_agents=num_agents,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio,
        value_loss_coef=value_loss_coef, entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm, hidden_dims=hidden_dims, device=device
    )
    
    buffer = RolloutBuffer(buffer_size, obs_dim, num_agents, num_actions, device)
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    kl_divs = []
    clipped_ratios = []
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Training loop
    pbar = tqdm(range(total_episodes), desc="Training MAPG")
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
            
            # Take step
            next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
            done = terminated or truncated
            
            # Store rewards
            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])
            
            # Store experience
            buffer.add(
                obs=obs_tensor.squeeze(0),
                actions=actions_tensor.squeeze(0),
                rewards=torch.FloatTensor(rewards).to(device),
                values=values.squeeze(0),
                log_probs=log_probs.squeeze(0),
                dones=torch.tensor(done, device=device)
            )
            
            obs = next_obs
            step_count += 1
            
            if done:
                break
        
        # Calculate episode return (discounted sum)
        episode_returns_agents = []
        for i in range(num_agents):
            agent_return = sum((gamma ** t) * r for t, r in enumerate(episode_rewards[i]))
            episode_returns_agents.append(agent_return)
        
        episode_return = sum(episode_returns_agents)
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        
        # Save best models
        if episode_return > best_return:
            best_return = episode_return
            # Copy shared value network to individual value networks for saving
            for i in range(agent.num_agents):
                agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
            
            # Save best individual agent files
            for i in range(agent.num_agents):
                torch.save({
                    'episode': episode,
                    'policy_state_dict': agent.policy_networks[i].state_dict(),
                    'value_state_dict': agent.value_networks[i].state_dict(),
                    'return': episode_return
                }, models_dir / f"best_agent_{i}.pt")
        
        # Update policy if buffer is full
        if buffer.size >= buffer_size:
            losses = agent.update(buffer, epochs=update_epochs)
            
            if losses:
                policy_losses.append(losses['policy_loss'])
                value_losses.append(losses['value_loss'])
                entropy_losses.append(losses['entropy_loss'])
                kl_divs.append(losses['kl_div'])
                clipped_ratios.append(losses['clipped_ratio'])
            
            buffer.clear()
            
            # Clear GPU cache to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update progress bar
        recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
        avg_return = np.mean(recent_returns)
        
        recent_policy_loss = np.mean(policy_losses[-10:]) if policy_losses else 0.0
        recent_kl_div = np.mean(kl_divs[-10:]) if kl_divs else 0.0
        
        pbar.set_postfix({
            'Episode': episode,
            'Return': f"{episode_return:.2f}",
            'Avg_Return': f"{avg_return:.2f}",
            'Policy_Loss': f"{recent_policy_loss:.3f}",
            'KL_Div': f"{recent_kl_div:.3f}",
            'Buffer': buffer.size,
            'Steps': step_count
        })
        
        # Log to CSV
        current_lr = agent.policy_optimizers[0].param_groups[0]['lr']
        recent_value_loss = np.mean(value_losses[-10:]) if value_losses else 0.0
        recent_entropy_loss = np.mean(entropy_losses[-10:]) if entropy_losses else 0.0
        recent_clipped_ratio = np.mean(clipped_ratios[-10:]) if clipped_ratios else 0.0
        
        with open(logs_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [
                episode, step_count, f"{episode_return:.2f}",
                f"{episode_return:.2f}", f"{0:.2f}"  # return_mean and return_std (single episode)
            ] + [f"{r:.2f}" for r in episode_returns_agents] + [
                f"{recent_policy_loss:.4f}",
                f"{recent_value_loss:.4f}",
                f"{recent_entropy_loss:.4f}",
                f"{recent_kl_div:.4f}",
                f"{recent_clipped_ratio:.4f}",
                buffer.size,
                f"{current_lr:.6f}"
            ]
            writer.writerow(row)
        
        # Save models - MADQN style individual agent files
        if episode % save_interval == 0 and episode > 0:
            # Copy shared value network to individual value networks for saving
            for i in range(agent.num_agents):
                agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
            
            # Save individual agent files
            for i in range(agent.num_agents):
                torch.save({
                    'episode': episode,
                    'policy_state_dict': agent.policy_networks[i].state_dict(),
                    'value_state_dict': agent.value_networks[i].state_dict(),
                    'policy_optimizer_state_dict': agent.policy_optimizers[i].state_dict(),
                    'return': episode_return
                }, models_dir / f"checkpoint_agent_{i}_ep{episode}.pt")
    
    # Save final models - MADQN style individual agent files
    # Copy shared value network to individual value networks for saving
    for i in range(agent.num_agents):
        agent.value_networks[i].load_state_dict(agent.shared_value_network.state_dict())
    
    # Save individual agent files
    for i in range(agent.num_agents):
        torch.save({
            'episode': total_episodes,
            'policy_state_dict': agent.policy_networks[i].state_dict(),
            'value_state_dict': agent.value_networks[i].state_dict(),
            'policy_optimizer_state_dict': agent.policy_optimizers[i].state_dict()
        }, models_dir / f"final_agent_{i}.pt")
    
    # Training log is already saved during training loop in MADQN style
    
    # Create training plots with error handling
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    except Exception as e:
        print(f"Error creating plots: {e}")
        return run_dir, best_return
    
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
        axes[0, 1].plot(policy_losses, alpha=0.7, color='green')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    if kl_divs:
        axes[1, 0].plot(kl_divs, alpha=0.7, color='orange')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('KL Div')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Clipped ratio
    if clipped_ratios:
        axes[1, 1].plot(clipped_ratios, alpha=0.7, color='purple')
        axes[1, 1].set_title('Clipped Ratio')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)
    
    try:
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
    
    # Save simple training log
    training_log = {
        "total_training_time_seconds": float(total_training_time),
        "total_training_time_hours": float(total_training_time / 3600),
        "total_episodes": int(total_episodes),
        "best_return": float(np.max(episode_returns)) if episode_returns else 0.0
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
    
    # Store return values before cleanup
    return_agent = agent
    return_episode_returns = episode_returns
    
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
    
    return return_agent, return_episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPG on GridWorld25v0")
    parser.add_argument("--total_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--buffer_size", type=int, default=2048, help="Rollout buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="MAPG clip ratio")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update_epochs", type=int, default=4, help="Update epochs per batch")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--mode", type=str, default="mode_2", help="Environment mode")
    
    args = parser.parse_args()
    
    agent, returns = train_mapg(
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
        mode=args.mode
    )
