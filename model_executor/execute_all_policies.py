#!/usr/bin/env python3
"""
Execute All Policy Models - Comprehensive Evaluation Script

This script executes all specified policy models (LLM, ep250, ep500, ep1500, no expert)
with added policy noise for deterministic environments (mode_2).
Saves all results to a single CSV file with policy indexing.
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expert_rl import GridWorld25v0

# Network definitions (copied from original files to avoid import issues)


class GaussianNoise(nn.Module):
    """Gaussian noise layer for policy randomization"""
    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = float(sigma)
    
    def forward(self, x):
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


class ImprovedDQNNetwork(nn.Module):
    """Enhanced DQN network with dueling architecture option"""
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128],
                 dropout: float = 0.0, use_dueling: bool = True, use_noisy: bool = False):
        super().__init__()
        self.use_dueling = use_dueling
        
        # Build shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
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
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        shared_features = self.shared_layers(x)
        value = self.value_head(shared_features)
        return value


class SmallTabNet(nn.Module):
    """LLM Expert Model Architecture"""
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
        param_device = next(self.parameters()).device
        if x.device != param_device:
            x = x.to(param_device)
        x = self.noise(x)
        return self.mlp(x)


def load_llm_model(ckpt_path: Path, device: torch.device):
    """Load LLM expert model"""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = SmallTabNet(in_dim=18, hidden1=64, hidden2=64, out_dim=6, p_drop=0.0, noise=0.0).to(device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    mu = ckpt["mu"].to(device=device, dtype=torch.float32)
    std = ckpt["std"].to(device=device, dtype=torch.float32)
    return model, mu, std


def load_madqn_model(model_path: Path, device: torch.device, agent_idx: int):
    """Load MADQN model for a specific agent"""
    policy_path = model_path / f"policy_agent_{agent_idx}.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    
    # Load hyperparameters to reconstruct network
    config_path = model_path.parent / "configs" / "hparams.json"
    with open(config_path, 'r') as f:
        hparams = json.load(f)
    
    # Create network with same architecture
    network = ImprovedDQNNetwork(
        obs_dim=hparams['obs_dim'],
        num_actions=hparams['num_actions'],
        hidden_dims=hparams['hidden_dims'],
        dropout=hparams.get('dropout', 0.0),
        use_dueling=hparams.get('use_dueling', True),
        use_noisy=hparams.get('use_noisy', False)
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(policy_path, map_location=device)
    network.load_state_dict(checkpoint)
    network.eval()
    
    return network


def load_mapg_model(model_path: Path, device: torch.device, agent_idx: int):
    """Load MAPG model for a specific agent"""
    # Try different possible file names
    possible_names = [
        f"final_agent_{agent_idx}.pt",
        f"best_agent_{agent_idx}.pt", 
        f"policy_agent_{agent_idx}.pt"
    ]
    
    policy_path = None
    for name in possible_names:
        candidate_path = model_path / name
        if candidate_path.exists():
            policy_path = candidate_path
            break
    
    if policy_path is None:
        raise FileNotFoundError(f"Policy not found in {model_path} for agent {agent_idx}")
    
    # Load hyperparameters to reconstruct network
    config_path = model_path.parent / "configs" / "hparams.json"
    with open(config_path, 'r') as f:
        hparams = json.load(f)
    
    # Create policy network
    policy_network = PolicyNetwork(
        obs_dim=hparams['obs_dim'],
        num_actions=hparams['num_actions'],
        hidden_dims=hparams['hidden_dims']
    ).to(device)
    
    # Load policy weights - handle nested state_dict structure
    policy_checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    
    # Extract policy state_dict from nested structure
    if 'policy_state_dict' in policy_checkpoint:
        policy_state_dict = policy_checkpoint['policy_state_dict']
    else:
        policy_state_dict = policy_checkpoint
    
    policy_network.load_state_dict(policy_state_dict)
    policy_network.eval()
    
    return policy_network


def get_action_with_noise(model, obs, device, noise_std=0.1, model_type="policy"):
    """Get action with added noise for deterministic environments"""
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        if model_type == "llm":
            # For LLM model, we need to normalize obs first
            # This will be handled in the calling function
            logits = model(obs_tensor)
        elif model_type == "madqn":
            # For DQN, get Q-values
            q_values = model(obs_tensor)
            logits = q_values
        elif model_type == "policy":
            # For policy network, get logits from distribution
            policy_dist = model(obs_tensor)
            logits = policy_dist.logits
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add noise to logits for randomization
        noise = torch.randn_like(logits) * noise_std
        noisy_logits = logits + noise
        
        # Sample action from noisy logits
        action_probs = F.softmax(noisy_logits, dim=-1)
        # Ensure action_probs is 2D for multinomial
        if action_probs.dim() == 1:
            action_probs = action_probs.unsqueeze(0)
        action = torch.multinomial(action_probs, 1).item()
    
    return action


def execute_single_episode(models, model_type, device, gamma, max_steps=200, 
                          noise_std=0.1, normalize_obs=None):
    """Execute a single episode with the given models"""
    # Create environment (mode_2 for deterministic)
    env = GridWorld25v0(max_steps=max_steps, gamma=gamma, mode="mode_2")
    obs, info = env.reset()
    
    returns_per_agent = [[] for _ in range(4)]
    length = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and length < max_steps:
        # Compute actions for all agents
        actions = []
        for i in range(4):
            if model_type == "llm":
                # LLM model needs normalization
                obs_i = torch.tensor(obs[i], dtype=torch.float32, device=device)
                if normalize_obs is not None:
                    mu, std = normalize_obs
                    # Handle 2D mu/std tensors
                    if mu.dim() == 2:
                        mu = mu.squeeze(0)
                        std = std.squeeze(0)
                    obs_i = (obs_i - mu) / std
                obs_i = obs_i.unsqueeze(0)
                
                with torch.no_grad():
                    logits = models[0](obs_i)  # Same model for all agents
                    noise = torch.randn_like(logits) * noise_std
                    noisy_logits = logits + noise
                    action_probs = F.softmax(noisy_logits, dim=-1)
                    # Ensure action_probs is 2D for multinomial
                    if action_probs.dim() == 1:
                        action_probs = action_probs.unsqueeze(0)
                    action = torch.multinomial(action_probs, 1).item()
            else:
                # MADQN or MAPG models
                action = get_action_with_noise(models[i], obs[i], device, noise_std, model_type)
            
            actions.append(action)
        
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        for i in range(4):
            returns_per_agent[i].append(rewards[i])
        
        obs = next_obs
        length += 1
    
    # Calculate discounted returns
    agent_returns = []
    for i in range(4):
        discounted_return = sum((gamma ** t) * r for t, r in enumerate(returns_per_agent[i]))
        agent_returns.append(discounted_return)
    
    env.close()
    
    return {
        "episode_return": float(sum(agent_returns)),
        "episode_length": int(length),
        "agent_returns": [float(x) for x in agent_returns],
        "total_rewards": [float(sum(rs)) for rs in returns_per_agent],
        "terminated": bool(terminated),
        "truncated": bool(truncated)
    }


def main():
    parser = argparse.ArgumentParser(description="Execute all policy models")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs per policy")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Policy noise standard deviation")
    parser.add_argument("--output_file", type=str, default="all_policies_execution_results.csv", 
                       help="Output CSV filename")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define model paths and configurations (using absolute paths)
    project_root = Path(__file__).parent.parent
    model_configs = [
        {
            "name": "llm",
            "path": project_root / "lbf_llm/lbf_small.pt",
            "type": "llm",
            "algorithm": "llm_policy"
        },
        {
            "name": "llm_expert",
            "path": project_root / "mapg_expert_rl/results/run_69/models",
            "type": "policy",
            "algorithm": "mapg_llm_expert"
        },
        {
            "name": "no_expert",
            "path": project_root / "mapg_expert_rl/results/run_9/models",
            "type": "policy",
            "algorithm": "mapg_no_expert"
        },
        {
            "name": "ep250",
            "path": project_root / "mapg_expert_rl/results/run_49/models",
            "type": "policy", 
            "algorithm": "mapg_ep250"
        },
        {
            "name": "ep500",
            "path": project_root / "mapg_expert_rl/results/run_50/models",
            "type": "policy",
            "algorithm": "mapg_ep500"
        },
        {
            "name": "ep1500",
            "path": project_root / "mapg_expert_rl/results/run_52/models",
            "type": "policy",
            "algorithm": "mapg_ep1500"
        }
    ]
    
    all_results = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Executing {config['name'].upper()} policy...")
        print(f"{'='*60}")
        
        try:
            if config["type"] == "llm":
                # Load LLM model
                model, mu, std = load_llm_model(config["path"], device)
                models = [model]  # Same model for all agents
                normalize_obs = (mu, std)
            else:
                # Load MAPG models
                models = []
                for agent_idx in range(4):
                    model = load_mapg_model(config["path"], device, agent_idx)
                    models.append(model)
                normalize_obs = None
            
            # Execute multiple runs
            policy_results = []
            pbar = tqdm(range(args.num_runs), desc=f"Running {config['name']}")
            
            for run_idx in pbar:
                # Set random seed for this run (for environment determinism)
                run_seed = run_idx + 1
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                random.seed(run_seed)
                
                result = execute_single_episode(
                    models=models,
                    model_type=config["type"],
                    device=device,
                    gamma=args.gamma,
                    max_steps=args.max_steps,
                    noise_std=args.noise_std,
                    normalize_obs=normalize_obs
                )
                
                # Add metadata
                result.update({
                    "policy_name": config["name"],
                    "algorithm": config["algorithm"],
                    "run_idx": run_idx,
                    "run_seed": run_seed,
                    "model_path": str(config["path"]),
                    "noise_std": args.noise_std
                })
                
                policy_results.append(result)
                all_results.append(result)
                
                pbar.set_postfix({
                    "Return": f"{result['episode_return']:.3f}",
                    "Length": result['episode_length']
                })
            
            # Print summary for this policy
            returns = [r['episode_return'] for r in policy_results]
            print(f"\n{config['name'].upper()} Summary:")
            print(f"  Mean Return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
            print(f"  Min/Max: {np.min(returns):.3f} / {np.max(returns):.3f}")
            print(f"  Mean Length: {np.mean([r['episode_length'] for r in policy_results]):.1f}")
            
        except Exception as e:
            print(f"Error executing {config['name']}: {e}")
            continue
    
    # Save all results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = Path(args.output_file)
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"All results saved to: {output_path}")
        print(f"Total runs: {len(all_results)}")
        print(f"{'='*60}")
        
        # Print overall summary
        print("\nOVERALL SUMMARY:")
        for policy_name in df['policy_name'].unique():
            policy_data = df[df['policy_name'] == policy_name]
            returns = policy_data['episode_return'].values
            print(f"{policy_name.upper():>10}: {np.mean(returns):.3f} ± {np.std(returns):.3f} "
                  f"(n={len(returns)})")
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
