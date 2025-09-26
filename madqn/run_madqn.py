#!/usr/bin/env python3
"""
MADQN Training Runner Script

This script provides an easy way to run MADQN training with different configurations.
You can modify the parameters below or use command-line arguments to customize training.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from madqn.train_madqn import train_madqn

def main():
    parser = argparse.ArgumentParser(description="Run MADQN training with configurable parameters")
    
    # Environment parameters
    parser.add_argument("--env_id", type=str, default="expert_rl/GridWorld25v0", 
                       help="Environment ID")
    parser.add_argument("--max_steps_per_ep", type=int, default=200,
                       help="Maximum steps per episode")
    
    # Training parameters
    parser.add_argument("--total_episodes", type=int, default=2000,
                       help="Total number of training episodes")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97,
                       help="Discount factor")
    
    # Network parameters
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[128, 128],
                       help="Hidden layer dimensions")
    parser.add_argument("--use_dueling", action='store_true', default=True,
                       help="Use dueling DQN architecture")
    parser.add_argument("--use_double_dqn", action='store_true', default=True,
                       help="Use double DQN")
    
    # Buffer parameters
    parser.add_argument("--buffer_capacity", type=int, default=100000,
                       help="Replay buffer capacity")
    parser.add_argument("--min_buffer_before_training", type=int, default=1000,
                       help="Minimum buffer size before training starts")
    parser.add_argument("--use_prioritized_replay", action='store_true', default=True,
                       help="Use prioritized experience replay")
    
    # Exploration parameters
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Initial epsilon value")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                       help="Final epsilon value")
    parser.add_argument("--epsilon_decay_episodes", type=int, default=300,
                       help="Episodes for epsilon decay")
    
    # Training stability parameters
    parser.add_argument("--update_every", type=int, default=3,
                       help="Update frequency (steps)")
    parser.add_argument("--soft_update_tau", type=float, default=0.005,
                       help="Soft update parameter for target networks")
    parser.add_argument("--grad_clip", type=float, default=10.0,
                       help="Gradient clipping value")
    parser.add_argument("--lr_decay_factor", type=float, default=0.995,
                       help="Learning rate decay factor")
    
    # Other parameters
    parser.add_argument("--normalize_rewards", action='store_true', default=True,
                       help="Normalize rewards")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Convert hidden_dims to list if it's not already
    if isinstance(args.hidden_dims, int):
        args.hidden_dims = [args.hidden_dims]
    
    # Set device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("MADQN TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Total Episodes: {args.total_episodes}")
    print(f"Max Steps per Episode: {args.max_steps_per_ep}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dimensions: {args.hidden_dims}")
    print(f"Dueling DQN: {args.use_dueling}")
    print(f"Double DQN: {args.use_double_dqn}")
    print(f"Prioritized Replay: {args.use_prioritized_replay}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} ({args.epsilon_decay_episodes} episodes)")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Run training
    train_madqn(
        env_id=args.env_id,
        total_episodes=args.total_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        soft_update_tau=args.soft_update_tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        min_buffer_before_training=args.min_buffer_before_training,
        update_every=args.update_every,
        use_prioritized_replay=args.use_prioritized_replay,
        normalize_rewards=args.normalize_rewards,
        use_dueling=args.use_dueling,
        use_double_dqn=args.use_double_dqn,
        hidden_dims=args.hidden_dims,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=device
    )

if __name__ == "__main__":
    main()
