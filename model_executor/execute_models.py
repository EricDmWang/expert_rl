#!/usr/bin/env python3
"""
Model Execution Script for Evaluating Trained Multi-Agent RL Models

This script can execute trained models from different algorithms (MADQN, QMIX, PPO)
and evaluate their performance across multiple runs with different random seeds.
"""

import os
import sys
import json
import argparse
import random
import time
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expert_rl import GridWorld25v0


class ImprovedDQNNetwork(nn.Module):
    """Enhanced DQN network with dueling architecture option (from MADQN)"""
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


class QNetwork(nn.Module):
    """Individual Q-network for each agent (from QMIX)"""
    
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
        
        # Initialize weights with better strategy
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
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        y = torch.bmm(hidden, w_final) + v
        
        return y.view(batch_size, 1)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO (from PPO)"""
    
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        # Shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Initialize weights conservatively
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor):
        shared_features = self.shared(x)
        
        # Policy logits
        action_logits = self.actor(shared_features)
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        
        # Value estimate
        value = self.critic(shared_features)
        
        return action_probs, value, action_logits


class ModelExecutor:
    """Executor for different types of trained models"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load hyperparameters
        config_path = self.model_path / "configs" / "hparams.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.hparams = json.load(f)
        
        # Determine model type from path
        if "madqn" in str(self.model_path):
            self.model_type = "madqn"
        elif "qmix" in str(self.model_path):
            self.model_type = "qmix"
        elif "mapg" in str(self.model_path):
            self.model_type = "mapg"
        else:
            raise ValueError(f"Unknown model type from path: {self.model_path}")
        
        print(f"Loading {self.model_type} model from: {self.model_path}")
        print(f"Hyperparameters: {self.hparams}")
    
    def load_models(self):
        """Load the trained models based on type"""
        if self.model_type == "madqn":
            return self._load_madqn_models()
        elif self.model_type == "qmix":
            return self._load_qmix_models()
        elif self.model_type == "mapg":
            return self._load_mapg_models()
    
    def _load_madqn_models(self):
        """Load MADQN models"""
        models = []
        
        # Load individual agent models
        for i in range(self.hparams['num_agents']):
            # model_path = self.model_path / "models" / f"best_agent_{i}.pt"
            model_path = self.model_path / "models" / f"final_agent_{i}.pt"
            if not model_path.exists():
                # Try checkpoint models
                model_path = self.model_path / "models" / f"checkpoint_agent_{i}_ep{self.hparams.get('total_episodes', 1000)}.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"MADQN model not found for agent {i}: {model_path}")
            
            # Create network
            model = ImprovedDQNNetwork(
                obs_dim=self.hparams['obs_dim'],
                num_actions=self.hparams['num_actions'],
                hidden_dims=self.hparams.get('hidden_dims', [128, 128]),
                use_dueling=self.hparams.get('use_dueling', True)
            ).to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
        
        return models
    
    def _load_qmix_models(self):
        """Load QMIX models"""
        model_path = self.model_path / "models" / "final_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"QMIX model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create Q-networks
        q_networks = nn.ModuleList([
            QNetwork(
                obs_dim=self.hparams['obs_dim'],
                num_actions=self.hparams['num_actions'],
                hidden_dims=self.hparams.get('hidden_dims', [128, 128]),
                use_dueling=True
            ).to(self.device)
            for _ in range(self.hparams['num_agents'])
        ])
        
        # Create mixer
        mixer = QMixer(
            state_dim=self.hparams['state_dim'],
            num_agents=self.hparams['num_agents'],
            mixing_embed_dim=self.hparams.get('mixing_embed_dim', 32),
            hypernet_embed_dim=self.hparams.get('hypernet_embed_dim', 64)
        ).to(self.device)
        
        # Load weights
        q_networks.load_state_dict(checkpoint['q_networks'])
        mixer.load_state_dict(checkpoint['mixer'])
        
        q_networks.eval()
        mixer.eval()
        
        return q_networks, mixer
    
    def _load_mapg_models(self):
        """Load MAPG models with CTDE architecture - MADQN style individual agent files"""
        # Import the networks
        from mapg_expert_rl.train_mapg import PolicyNetwork, ValueNetwork
        
        obs_dim = self.hparams['obs_dim']
        num_actions = self.hparams['num_actions']
        num_agents = self.hparams['num_agents']
        hidden_dims = self.hparams.get('hidden_dims', [128, 128])
        
        models = []
        for i in range(num_agents):
            # Try to load individual agent file
            agent_path = self.model_path / "models" / f"final_agent_{i}.pt"
            if not agent_path.exists():
                raise FileNotFoundError(f"MAPG agent {i} model not found: {agent_path}")
            
            # Load agent checkpoint
            checkpoint = torch.load(agent_path, map_location=self.device)
            
            # Create policy network
            policy_net = PolicyNetwork(obs_dim, num_actions, hidden_dims).to(self.device)
            policy_net.load_state_dict(checkpoint['policy_state_dict'])
            policy_net.eval()
            
            # Create value network
            value_net = ValueNetwork(obs_dim, hidden_dims).to(self.device)
            value_net.load_state_dict(checkpoint['value_state_dict'])
            value_net.eval()
            
            models.append((policy_net, value_net))
        
        return models
    
    def execute_single_episode(self, models, env, max_steps: int = 200, render: bool = False) -> Dict[str, Any]:
        """Execute a single episode with the given models"""
        obs, info = env.reset()
        episode_rewards = [[] for _ in range(self.hparams['num_agents'])]
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Get actions from models
            if self.model_type == "madqn":
                actions = self._get_madqn_actions(models, obs)
            elif self.model_type == "qmix":
                actions = self._get_qmix_actions(models, obs)
            elif self.model_type == "mapg":
                actions = self._get_mapg_actions(models, obs)
            
            # Take step
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store rewards
            for i in range(self.hparams['num_agents']):
                episode_rewards[i].append(rewards[i])
            
            # Render if requested
            if render:
                env.render()
            
            obs = next_obs
            episode_length += 1
        
        # Calculate episode returns
        gamma = self.hparams.get('gamma', 0.95)
        episode_returns = []
        for i in range(self.hparams['num_agents']):
            agent_return = sum((gamma ** t) * r for t, r in enumerate(episode_rewards[i]))
            episode_returns.append(agent_return)
        
        return {
            'episode_return': float(sum(episode_returns)),
            'episode_length': int(episode_length),
            'agent_returns': [float(x) for x in episode_returns],  # Convert to Python floats
            'total_rewards': [float(sum(rewards)) for rewards in episode_rewards],  # Convert to Python floats
            'terminated': bool(terminated),
            'truncated': bool(truncated)
        }
    
    def _get_madqn_actions(self, models, obs):
        """Get actions from MADQN models"""
        actions = []
        for i, model in enumerate(models):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=1).item()
            actions.append(action)
        return actions
    
    def _get_qmix_actions(self, models, obs):
        """Get actions from QMIX models"""
        q_networks, mixer = models
        actions = []
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for i in range(self.hparams['num_agents']):
                q_values = q_networks[i](obs_tensor[:, i])
                action = q_values.argmax(dim=1).item()
                actions.append(action)
        
        return actions
    
    def _get_mapg_actions(self, models, obs):
        """Get actions from MAPG models with CTDE architecture"""
        actions = []
        
        for i, (policy_net, value_net) in enumerate(models):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_dist = policy_net(obs_tensor)
                action = policy_dist.sample().item()
            actions.append(action)
        
        return actions


def execute_single_model(model_path: str, num_runs: int = 30, max_steps: int = 200, 
                        render_first_n: int = 5):
    """Execute a single trained model and save results"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create executor
    executor = ModelExecutor(model_path, device)
    
    # Load models
    models = executor.load_models()
    
    # Get the training seed from hyperparameters
    training_seed = executor.hparams.get('seed', 1)
    print(f"Using training seed: {training_seed}")
    
    # Create results directory organized by algorithm
    model_path_obj = Path(model_path)
    # Extract algorithm name from path (e.g., "madqn" from "madqn/results/run_13")
    path_parts = model_path_obj.parts
    algorithm_name = path_parts[0] if len(path_parts) > 0 else "unknown"  # e.g., "madqn", "qmix_expert_rl", "mapg_expert_rl"
    run_name = model_path_obj.name  # e.g., "run_13"
    
    # Create algorithm-specific folder structure
    results_dir = Path("model_executor/results")
    algorithm_dir = results_dir / algorithm_name  # e.g., "model_executor/results/madqn"
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next execution number
    existing_executions = list(algorithm_dir.glob('execution_*'))
    next_exec_num = len(existing_executions)
    execution_dir = algorithm_dir / f"execution_{next_exec_num}"
    execution_dir.mkdir(exist_ok=True)
    
    render_dir = execution_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    
    print(f"Results will be saved to: {execution_dir}")
    
    # Execute episodes using only the training seed
    results = []
    pbar = tqdm(range(num_runs), desc=f"Executing {executor.model_type.upper()} - {run_name}")
    
    for run_idx in pbar:
        # Use the same training seed for all runs
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        random.seed(training_seed)
        
        # Create environment with training seed
        env = GridWorld25v0(
            max_steps=max_steps, 
            gamma=executor.hparams.get('gamma', 0.95), 
            mode="mode_2",
            seed=training_seed  # Use the training seed for environment initialization
        )
        
        # Determine if we should render this run
        should_render = run_idx < render_first_n
        
        if should_render:
            # Create unique render directory for this run
            run_render_dir = render_dir / f"run_{run_idx}"
            run_render_dir.mkdir(exist_ok=True)
            
            # Set render directory for this environment
            env._render_out_dir = str(run_render_dir)
            env._render_frame_idx = 0
        
        # Execute episode
        result = executor.execute_single_episode(models, env, max_steps, render=should_render)
        
        if should_render:
            # Generate GIF for this run
            try:
                env.save_animation()
                # Also copy GIF to main execution directory with run number
                source_gif = run_render_dir / "animation.gif"
                if source_gif.exists():
                    dest_gif = execution_dir / f"run_{run_idx}_animation.gif"
                    import shutil
                    shutil.copy2(source_gif, dest_gif)
            except Exception:
                pass
        
        # Close environment to free resources
        env.close()
        
        # Add metadata
        result.update({
            'run': run_idx,
            'seed': training_seed,
            'model_type': executor.model_type,
            'algorithm': algorithm_name,
            'model_path': str(model_path)
        })
        
        results.append(result)
        
        # Update progress
        avg_return = np.mean([r['episode_return'] for r in results])
        pbar.set_postfix({
            'Run': run_idx,
            'Return': f"{result['episode_return']:.3f}",
            'Avg_Return': f"{avg_return:.3f}",
            'Length': result['episode_length']
        })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = execution_dir / f"execution_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Create detailed execution information
    execution_info = {
        'execution_metadata': {
            'execution_id': next_exec_num,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'num_runs': num_runs,
            'max_steps_per_run': max_steps,
            'render_first_n_runs': render_first_n,
            'training_seed': training_seed,
            'seeds_used': [training_seed] * num_runs  # All runs use the same training seed
        },
        'model_information': {
            'model_path': str(model_path),
            'model_type': executor.model_type,
            'algorithm_name': algorithm_name,
            'run_name': run_name,
            'hyperparameters': executor.hparams
        },
        'environment_information': {
            'env_name': 'expert_rl/GridWorld25v0',
            'env_class': 'GridWorld25v0',
            'num_agents': executor.hparams['num_agents'],
            'obs_dim': executor.hparams['obs_dim'],
            'num_actions': executor.hparams['num_actions'],
            'gamma': executor.hparams.get('gamma', 0.95),
            'max_steps': max_steps
        },
        'execution_results': {
            'mean_episode_return': float(df['episode_return'].mean()),
            'std_episode_return': float(df['episode_return'].std()),
            'min_episode_return': float(df['episode_return'].min()),
            'max_episode_return': float(df['episode_return'].max()),
            'mean_episode_length': float(df['episode_length'].mean()),
            'std_episode_length': float(df['episode_length'].std()),
            'success_rate': float((df['episode_return'] > 0).mean()),
            'terminated_rate': float(df['terminated'].mean()),
            'truncated_rate': float(df['truncated'].mean()),
            'agent_returns_mean': [float(df['agent_returns'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(lambda x: x[i]).mean()) for i in range(executor.hparams['num_agents'])],
            'agent_returns_std': [float(df['agent_returns'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(lambda x: x[i]).std()) for i in range(executor.hparams['num_agents'])]
        },
        'files_created': {
            'csv_results': str(csv_path),
            'summary_json': str(execution_dir / "execution_info.json"),
            'plots': str(execution_dir / "execution_plots.png") if render_first_n > 0 else None,
            'gif_animations': [str(execution_dir / f"run_{i}_animation.gif") for i in range(min(render_first_n, num_runs))] if render_first_n > 0 else []
        }
    }
    
    # Save detailed execution information
    execution_info_path = execution_dir / "execution_info.json"
    with open(execution_info_path, 'w') as f:
        json.dump(execution_info, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode returns
    axes[0, 0].plot(df['episode_return'], alpha=0.7)
    axes[0, 0].axhline(y=df['episode_return'].mean(), color='red', linestyle='--', label=f'Mean: {df["episode_return"].mean():.3f}')
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Run')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(df['episode_length'], alpha=0.7)
    axes[0, 1].axhline(y=df['episode_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["episode_length"].mean():.1f}')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Run')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Return distribution
    axes[1, 0].hist(df['episode_return'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=df['episode_return'].mean(), color='red', linestyle='--', label=f'Mean: {df["episode_return"].mean():.3f}')
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Agent returns
    for i in range(executor.hparams['num_agents']):
        agent_returns = df['agent_returns'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).apply(lambda x: x[i])
        axes[1, 1].plot(agent_returns, alpha=0.7, label=f'Agent {i}')
    axes[1, 1].set_title('Individual Agent Returns')
    axes[1, 1].set_xlabel('Run')
    axes[1, 1].set_ylabel('Return')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(execution_dir / "execution_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\nExecution completed for {run_name}! Results saved to: {execution_dir}")
    print(f"Mean Episode Return: {execution_info['execution_results']['mean_episode_return']:.3f} ± {execution_info['execution_results']['std_episode_return']:.3f}")
    print(f"Mean Episode Length: {execution_info['execution_results']['mean_episode_length']:.1f} ± {execution_info['execution_results']['std_episode_length']:.1f}")
    print(f"Success Rate (Return > 0): {execution_info['execution_results']['success_rate']:.1%}")
    print(f"Terminated Rate: {execution_info['execution_results']['terminated_rate']:.1%}")
    print(f"Truncated Rate: {execution_info['execution_results']['truncated_rate']:.1%}")
    
    return execution_info


def execute_models(model_paths: List[str], num_runs: int = 30, max_steps: int = 200, 
                  render_first_n: int = 5):
    """Execute multiple trained models and save results separately"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert single path to list for uniform handling
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    print(f"Executing {len(model_paths)} models...")
    
    all_results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\n{'='*60}")
        print(f"Executing model {i+1}/{len(model_paths)}: {model_path}")
        print(f"{'='*60}")
        
        try:
            result = execute_single_model(
                model_path=model_path,
                num_runs=num_runs,
                max_steps=max_steps,
                render_first_n=render_first_n
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error executing {model_path}: {e}")
            continue
    
    # Print overall summary
    print(f"\n{'='*60}")
    print(f"BATCH EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully executed: {len(all_results)}/{len(model_paths)} models")
    
    for result in all_results:
        model_path = result['model_information']['model_path']
        run_name = result['model_information']['run_name']
        mean_return = result['execution_results']['mean_episode_return']
        std_return = result['execution_results']['std_episode_return']
        print(f"{run_name}: {mean_return:.3f} ± {std_return:.3f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute trained multi-agent RL models")
    parser.add_argument("--model_paths", nargs='+', default=["madqn/results/run_13","madqn/results/run_24","madqn/results/run_25",], 
                       help="List of model paths (e.g., madqn/results/run_13 mapg_expert_rl/results/run_6)")
    # parser.add_argument("--model_paths", nargs='+', default=["mapg_expert_rl/results/run_9","mapg_expert_rl/results/run_10",\
    #     "mapg_expert_rl/results/run_11","mapg_expert_rl/results/run_12","mapg_expert_rl/results/run_13",], 
    #                    help="List of model paths (e.g., madqn/results/run_13 mapg_expert_rl/results/run_6)")
    
    parser.add_argument("--num_runs", type=int, default=1, help="Number of execution runs")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--render_first_n", type=int, default=1, help="Number of runs to render")
    
    args = parser.parse_args()
    
    execute_models(
        model_paths=args.model_paths,
        num_runs=args.num_runs,
        max_steps=args.max_steps,
        render_first_n=args.render_first_n
    )
