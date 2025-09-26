#!/usr/bin/env python3
"""
QMIX Training Script for Expert RL GridWorld25v0 Environment

Based on PyMARL2's QMIX implementation, adapted for the custom GridWorld25v0 environment.
Uses value decomposition with monotonic mixing for cooperative multi-agent learning.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expert_rl import GridWorld25v0


class ReplayBuffer:
    """Experience replay buffer for multi-agent Q-learning."""
    
    def __init__(self, capacity: int, obs_dim: int, num_agents: int, num_actions: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Buffer storage
        self.observations = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_observations = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.states = np.zeros((capacity, obs_dim * num_agents), dtype=np.float32)  # Global state
        self.next_states = np.zeros((capacity, obs_dim * num_agents), dtype=np.float32)
        
        self.size = 0
        self.ptr = 0
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
            next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray):
        """Add experience to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch_obs = torch.FloatTensor(self.observations[indices])
        batch_actions = torch.LongTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices])
        batch_next_obs = torch.FloatTensor(self.next_observations[indices])
        batch_dones = torch.BoolTensor(self.dones[indices])
        batch_states = torch.FloatTensor(self.states[indices])
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_states, batch_next_states


class QNetwork(nn.Module):
    """Individual Q-network for each agent with improved architecture from MADQN."""
    
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128], use_dueling: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.use_dueling = use_dueling
        
        # Build shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # LayerNorm for stability
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], num_actions)
            )
        else:
            # Standard DQN
            self.q_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], num_actions)
            )
        
        # Initialize weights with better strategy from MADQN
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)  # Very conservative initialization
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        
        if self.use_dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            # Combine value and advantage (using mean advantage for stability)
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            q_values = self.q_head(x)
        
        return q_values


class QMixer(nn.Module):
    """QMIX mixing network for value decomposition."""
    
    def __init__(self, state_dim: int, num_agents: int, mixing_embed_dim: int = 32, 
                 hypernet_embed_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetworks for generating weights with smaller initialization
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim * num_agents)
        )
        
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim)
        )
        
        # State-dependent bias
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # V(s) for final bias
        self.V = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
        # Initialize weights with smaller values for stability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)  # Very small gain
            nn.init.constant_(module.bias, 0)
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: [batch_size, num_agents] - Q-values for each agent
            states: [batch_size, state_dim] - Global states
        Returns:
            q_tot: [batch_size, 1] - Total Q-value
        """
        batch_size = agent_qs.size(0)
        
        # Generate weights and biases from state
        w1 = torch.abs(self.hyper_w_1(states))  # Ensure non-negative weights for monotonicity
        b1 = self.hyper_b_1(states)
        w_final = torch.abs(self.hyper_w_final(states))
        v = self.V(states)
        
        # Reshape for matrix multiplication
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        w_final = w_final.view(batch_size, self.mixing_embed_dim, 1)
        v = v.view(batch_size, 1, 1)
        
        # First layer
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        y = torch.bmm(hidden, w_final) + v
        
        return y.view(batch_size, 1)


class CentralValueNet(nn.Module):
    """Centralized critic for multi-agent coordination (from MADQN)"""
    def __init__(self, total_obs_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = total_obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights conservatively
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RewardNormalizer:
    """Running normalization for rewards (from MADQN)"""
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean = 0
        self.running_var = 1
        self.count = 0
    
    def update(self, rewards: np.ndarray):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean += delta * batch_count / total_count
        
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.running_var = M2 / total_count
        self.count = total_count
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return rewards
        
        std = np.sqrt(self.running_var + self.epsilon)
        return (rewards - self.running_mean) / std


class QMIXAgent:
    """QMIX agent with individual Q-networks, shared mixer, and centralized critic."""
    
    def __init__(self, obs_dim: int, num_actions: int, state_dim: int, num_agents: int,
                 lr: float = 5e-4, gamma: float = 0.95, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: int = 100000,
                 mixing_embed_dim: int = 32, hypernet_embed_dim: int = 64,
                 hidden_dims: List[int] = [128, 128], device: torch.device = None):
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Individual Q-networks for each agent (with dueling architecture)
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, num_actions, hidden_dims, use_dueling=True).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Target networks
        self.target_q_networks = nn.ModuleList([
            QNetwork(obs_dim, num_actions, hidden_dims, use_dueling=True).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Copy initial weights to target networks
        for i in range(num_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # QMIX mixer
        self.mixer = QMixer(state_dim, num_agents, mixing_embed_dim, hypernet_embed_dim).to(self.device)
        self.target_mixer = QMixer(state_dim, num_agents, mixing_embed_dim, hypernet_embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Centralized value network (from MADQN)
        self.central_value_net = CentralValueNet(state_dim, hidden_dims=[256, 256, 128]).to(self.device)
        self.target_central_value_net = CentralValueNet(state_dim, hidden_dims=[256, 256, 128]).to(self.device)
        self.target_central_value_net.load_state_dict(self.central_value_net.state_dict())
        
        # Reward normalizer
        self.reward_normalizer = RewardNormalizer(gamma=gamma)
        
        # Optimizers with better settings from MADQN
        self.qmix_optimizer = optim.AdamW(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()),
            lr=lr, weight_decay=1e-5, eps=1e-8
        )
        
        self.central_optimizer = optim.AdamW(
            self.central_value_net.parameters(),
            lr=lr, weight_decay=1e-5, eps=1e-8
        )
        
        # Learning rate schedulers for stability
        self.qmix_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.qmix_optimizer, mode='min', factor=0.5, patience=20
        )
        
        self.central_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.central_optimizer, mode='min', factor=0.5, patience=20
        )
        
        self.update_count = 0
    
    def select_actions(self, observations: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """Select actions using epsilon-greedy policy."""
        batch_size = observations.size(0)
        actions = torch.zeros((batch_size, self.num_agents), dtype=torch.long, device=self.device)
        
        for i in range(self.num_agents):
            if not evaluate and random.random() < self.epsilon:
                # Random action
                actions[:, i] = torch.randint(0, self.num_actions, (batch_size,), device=self.device)
            else:
                # Greedy action
                with torch.no_grad():
                    q_values = self.q_networks[i](observations[:, i])
                    actions[:, i] = q_values.argmax(dim=-1)
        
        return actions
    
    def get_q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q-values for given observations and actions."""
        q_values = torch.zeros((observations.size(0), self.num_agents), device=self.device)
        
        for i in range(self.num_agents):
            agent_q_values = self.q_networks[i](observations[:, i])
            q_values[:, i] = agent_q_values.gather(1, actions[:, i:i+1]).squeeze(1)
        
        return q_values
    
    def get_target_q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get target Q-values for given observations and actions."""
        q_values = torch.zeros((observations.size(0), self.num_agents), device=self.device)
        
        for i in range(self.num_agents):
            with torch.no_grad():
                agent_q_values = self.target_q_networks[i](observations[:, i])
                q_values[:, i] = agent_q_values.gather(1, actions[:, i:i+1]).squeeze(1)
        
        return q_values
    
    def update(self, batch: Tuple[torch.Tensor, ...], target_update_interval: int = 200):
        """Update Q-networks, mixer, and centralized critic with enhanced stability measures."""
        obs, actions, rewards, next_obs, dones, states, next_states = [t.to(self.device) for t in batch]
        
        batch_size = obs.size(0)
        
        # Normalize rewards using reward normalizer
        rewards_np = rewards.cpu().numpy()
        self.reward_normalizer.update(rewards_np.flatten())
        normalized_rewards = torch.FloatTensor(
            self.reward_normalizer.normalize(rewards_np.flatten()).reshape(rewards_np.shape)
        ).to(self.device)
        
        # Current Q-values
        current_q_values = self.get_q_values(obs, actions)
        current_q_total = self.mixer(current_q_values, states)
        
        # Centralized value for additional supervision
        current_central_value = self.central_value_net(states)
        
        # Target Q-values
        with torch.no_grad():
            next_actions = self.select_actions(next_obs, evaluate=True)
            next_q_values = self.get_target_q_values(next_obs, next_actions)
            next_q_total = self.target_mixer(next_q_values, next_states)
            
            # Centralized target value
            next_central_value = self.target_central_value_net(next_states)
            
            # TD targets with normalized rewards
            rewards_sum = normalized_rewards.sum(dim=1, keepdim=True)  # Sum rewards across agents
            qmix_targets = rewards_sum + self.gamma * (1 - dones.float().unsqueeze(1)) * next_q_total
            
            # Centralized value targets
            central_targets = rewards_sum + self.gamma * (1 - dones.float().unsqueeze(1)) * next_central_value
            
            # Clip targets to prevent explosion
            qmix_targets = torch.clamp(qmix_targets, -10.0, 10.0)
            central_targets = torch.clamp(central_targets, -10.0, 10.0)
        
        # Compute losses with Huber loss for stability
        qmix_td_error = current_q_total - qmix_targets
        qmix_loss = F.smooth_l1_loss(current_q_total, qmix_targets)
        
        central_td_error = current_central_value - central_targets
        central_loss = F.smooth_l1_loss(current_central_value, central_targets)
        
        # Check for NaN or infinite values
        if torch.isnan(qmix_loss) or torch.isinf(qmix_loss) or torch.isnan(central_loss) or torch.isinf(central_loss):
            print(f"Warning: Invalid loss detected: QMIX={qmix_loss.item()}, Central={central_loss.item()}")
            return {
                'loss': 0.0,
                'central_loss': 0.0,
                'td_error': 0.0,
                'epsilon': self.epsilon,
                'q_total': 0.0,
                'grad_norm': 0.0,
                'lr': self.qmix_optimizer.param_groups[0]['lr']
            }
        
        # Optimize QMIX networks
        self.qmix_optimizer.zero_grad(set_to_none=True)
        qmix_loss.backward()
        
        # Compute gradient norms before clipping
        qmix_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()), 
            max_norm=5.0  # Conservative clipping from MADQN
        )
        
        self.qmix_optimizer.step()
        self.qmix_scheduler.step(qmix_loss)
        
        # Optimize centralized value network
        self.central_optimizer.zero_grad(set_to_none=True)
        central_loss.backward()
        
        central_norm = torch.nn.utils.clip_grad_norm_(
            self.central_value_net.parameters(), 
            max_norm=5.0
        )
        
        self.central_optimizer.step()
        self.central_scheduler.step(central_loss)
        
        # Update epsilon more slowly (from MADQN)
        if self.update_count % 2000 == 0:  # Much slower epsilon decay
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon - (1.0 - self.epsilon_end) / (self.epsilon_decay * 4))
        
        # Update target networks with soft updates (from MADQN)
        if self.update_count % target_update_interval == 0:
            tau = 0.005  # Conservative soft update parameter
            for i in range(self.num_agents):
                for target_param, param in zip(self.target_q_networks[i].parameters(), 
                                             self.q_networks[i].parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            for target_param, param in zip(self.target_mixer.parameters(), 
                                         self.mixer.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            for target_param, param in zip(self.target_central_value_net.parameters(), 
                                         self.central_value_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        self.update_count += 1
        
        return {
            'loss': qmix_loss.item(),
            'central_loss': central_loss.item(),
            'td_error': qmix_td_error.abs().mean().item(),
            'epsilon': self.epsilon,
            'q_total': current_q_total.mean().item(),
            'grad_norm': qmix_norm.item() if hasattr(qmix_norm, 'item') else float(qmix_norm),
            'lr': self.qmix_optimizer.param_groups[0]['lr']
        }


def train_qmix(env_id: str = "expert_rl/GridWorld25v0", total_episodes: int = 1000,
               max_steps_per_ep: int = 200, buffer_capacity: int = 100000,
               batch_size: int = 128, lr: float = 1e-4, gamma: float = 0.95,  # Even smaller LR
               target_update_interval: int = 500, train_frequency: int = 8,
               min_buffer_size: int = 20000, save_interval: int = 100,
               mixing_embed_dim: int = 64, hypernet_embed_dim: int = 128):
    """Train QMIX on the GridWorld25v0 environment."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld25v0(max_steps=max_steps_per_ep, gamma=gamma, mode="mode_2",seed=1)
    obs, info = env.reset()
    
    obs_dim = obs.shape[1]  # 18
    num_actions = 6
    num_agents = 4
    state_dim = obs_dim * num_agents  # Global state is concatenated observations
    
    # Create results directory
    results_dir = Path("qmix_expert_rl/results")
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
        'buffer_capacity': buffer_capacity,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'target_update_interval': target_update_interval,
        'train_frequency': train_frequency,
        'min_buffer_size': min_buffer_size,
        'mixing_embed_dim': mixing_embed_dim,
        'hypernet_embed_dim': hypernet_embed_dim,
        'obs_dim': obs_dim,
        'num_actions': num_actions,
        'num_agents': num_agents,
        'state_dim': state_dim
    }
    
    with open(configs_dir / "hparams.json", 'w') as f:
        json.dump(hparams, f, indent=2)
    
    # Initialize agent and replay buffer
    agent = QMIXAgent(
        obs_dim=obs_dim, num_actions=num_actions, state_dim=state_dim, num_agents=num_agents,
        lr=lr, gamma=gamma, mixing_embed_dim=mixing_embed_dim, 
        hypernet_embed_dim=hypernet_embed_dim, device=device
    )
    
    replay_buffer = ReplayBuffer(buffer_capacity, obs_dim, num_agents, num_actions)
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    qmix_losses = []
    central_losses = []
    td_errors = []
    q_values = []
    epsilons = []
    grad_norms = []
    learning_rates = []
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Training loop
    pbar = tqdm(range(total_episodes), desc="Training QMIX")
    
    for episode in pbar:
        obs, info = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        episode_return = 0
        step_count = 0
        
        for step in range(max_steps_per_ep):
            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            state_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
            
            # Select actions
            actions_tensor = agent.select_actions(obs_tensor, evaluate=False)
            actions = actions_tensor.squeeze(0).cpu().numpy()
            
            # Take step
            next_obs, rewards, terminated, truncated, info = env.step(actions.tolist())
            done = terminated or truncated
            
            # Store rewards
            for i in range(num_agents):
                episode_rewards[i].append(rewards[i])
            
            # Convert next observations to tensors
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_obs.flatten()).unsqueeze(0).to(device)
            
            # Store experience
            replay_buffer.add(
                obs=obs, actions=actions, rewards=rewards, next_obs=next_obs,
                done=done, state=obs.flatten(), next_state=next_obs.flatten()
            )
            
            # Training
            if (replay_buffer.size >= min_buffer_size and 
                step_count % train_frequency == 0):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.update(batch, target_update_interval)
                
                qmix_losses.append(metrics['loss'])
                central_losses.append(metrics['central_loss'])
                td_errors.append(metrics['td_error'])
                q_values.append(metrics['q_total'])
                epsilons.append(metrics['epsilon'])
                grad_norms.append(metrics['grad_norm'])
                learning_rates.append(metrics['lr'])
            
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
        
        # Update progress bar
        recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
        avg_return = np.mean(recent_returns)
        
        # Get recent metrics for display
        recent_qmix_loss = np.mean(qmix_losses[-10:]) if qmix_losses else 0.0
        recent_central_loss = np.mean(central_losses[-10:]) if central_losses else 0.0
        recent_grad_norm = np.mean(grad_norms[-10:]) if grad_norms else 0.0
        current_lr = learning_rates[-1] if learning_rates else agent.qmix_optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            'Episode': episode,
            'Return': f"{episode_return:.2f}",
            'Avg_Return': f"{avg_return:.2f}",
            'Epsilon': f"{agent.epsilon:.3f}",
            'QMIX_Loss': f"{recent_qmix_loss:.3f}",
            'Central_Loss': f"{recent_central_loss:.3f}",
            'GradNorm': f"{recent_grad_norm:.2f}",
            'LR': f"{current_lr:.2e}",
            'Buffer': replay_buffer.size,
            'Steps': step_count
        })
        
        # Save models
        if episode % save_interval == 0 and episode > 0:
            torch.save({
                'q_networks': agent.q_networks.state_dict(),
                'mixer': agent.mixer.state_dict(),
                'central_value_net': agent.central_value_net.state_dict(),
                'qmix_optimizer': agent.qmix_optimizer.state_dict(),
                'central_optimizer': agent.central_optimizer.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon
            }, models_dir / f"checkpoint_ep{episode}.pt")
    
    # Save final models
    torch.save({
        'q_networks': agent.q_networks.state_dict(),
        'mixer': agent.mixer.state_dict(),
        'central_value_net': agent.central_value_net.state_dict(),
        'qmix_optimizer': agent.qmix_optimizer.state_dict(),
        'central_optimizer': agent.central_optimizer.state_dict(),
        'episode': total_episodes,
        'epsilon': agent.epsilon
    }, models_dir / "final_model.pt")
    
    # Save training log
    training_log = pd.DataFrame({
        'episode': range(len(episode_returns)),
        'episode_return': episode_returns,
        'episode_length': episode_lengths,
        'epsilon': epsilons[:len(episode_returns)] if epsilons else [agent.epsilon] * len(episode_returns),
        'qmix_loss': qmix_losses[:len(episode_returns)] if qmix_losses else [0.0] * len(episode_returns),
        'central_loss': central_losses[:len(episode_returns)] if central_losses else [0.0] * len(episode_returns),
        'td_error': td_errors[:len(episode_returns)] if td_errors else [0.0] * len(episode_returns),
        'q_value': q_values[:len(episode_returns)] if q_values else [0.0] * len(episode_returns),
        'grad_norm': grad_norms[:len(episode_returns)] if grad_norms else [0.0] * len(episode_returns),
        'learning_rate': learning_rates[:len(episode_returns)] if learning_rates else [agent.qmix_optimizer.param_groups[0]['lr']] * len(episode_returns)
    })
    
    training_log.to_csv(run_dir / "training_log.csv", index=False)
    
    # Create training plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode returns
    axes[0, 0].plot(episode_returns, alpha=0.3, color='blue')
    if len(episode_returns) > 20:
        window = min(20, len(episode_returns) // 10)
        moving_avg = pd.Series(episode_returns).rolling(window=window, center=True).mean()
        axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window})')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    if qmix_losses or central_losses:
        if qmix_losses:
            axes[0, 1].plot(qmix_losses, alpha=0.7, label='QMIX Loss', color='blue')
        if central_losses:
            axes[0, 1].plot(central_losses, alpha=0.7, label='Central Loss', color='red')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Q-values
    if q_values:
        axes[1, 0].plot(q_values, alpha=0.7)
        axes[1, 0].set_title('Average Q-Values')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Q-Value')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Norm and Learning Rate
    if grad_norms:
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(grad_norms, color='blue', alpha=0.7, label='Grad Norm')
        axes[1, 1].set_ylabel('Gradient Norm', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        axes[1, 1].set_yscale('log')
        
        if learning_rates:
            ax2.plot(learning_rates, color='red', alpha=0.7, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_yscale('log')
        
        axes[1, 1].set_title('Gradient Norm and Learning Rate')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(run_dir / "training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
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
    
    log_path = log_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    env.close()
    print(f"\nTraining completed! Results saved to: {run_dir}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"Final average return (last 100 episodes): {np.mean(episode_returns[-100:]):.2f}")
    
    return agent, episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QMIX on GridWorld25v0")
    parser.add_argument("--total_episodes", type=int, default=1000, help="Total training episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--target_update_interval", type=int, default=500, help="Target network update interval")
    parser.add_argument("--train_frequency", type=int, default=8, help="Training frequency")
    parser.add_argument("--min_buffer_size", type=int, default=20000, help="Minimum buffer size before training")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    
    args = parser.parse_args()
    
    agent, returns = train_qmix(
        total_episodes=args.total_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        train_frequency=args.train_frequency,
        min_buffer_size=args.min_buffer_size,
        save_interval=args.save_interval
    )
