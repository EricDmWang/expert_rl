#!/usr/bin/env python3
"""
QMIX Training Runner Script

This script provides an easy way to run QMIX training with different configurations.
You can modify the parameters below or use command-line arguments to customize training.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qmix_expert_rl.train_qmix import train_qmix

def main():
    parser = argparse.ArgumentParser(description="Run QMIX training with configurable parameters")
    
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
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95,
                       help="Discount factor")
    
    # Network parameters
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[128, 128],
                       help="Hidden layer dimensions")
    parser.add_argument("--mixing_embed_dim", type=int, default=64,
                       help="Mixing network embedding dimension")
    parser.add_argument("--hypernet_embed_dim", type=int, default=128,
                       help="Hypernetwork embedding dimension")
    parser.add_argument("--state_dim", type=int, default=72,
                       help="Global state dimension")
    
    # Buffer parameters
    parser.add_argument("--buffer_capacity", type=int, default=100000,
                       help="Replay buffer capacity")
    parser.add_argument("--min_buffer_size", type=int, default=20000,
                       help="Minimum buffer size before training starts")
    parser.add_argument("--train_frequency", type=int, default=8,
                       help="Training frequency (steps)")
    
    # Exploration parameters
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Initial epsilon value")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                       help="Final epsilon value")
    parser.add_argument("--epsilon_decay_episodes", type=int, default=500,
                       help="Episodes for epsilon decay")
    
    # Training stability parameters
    parser.add_argument("--target_update_interval", type=int, default=500,
                       help="Target network update interval")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                       help="Gradient clipping value")
    parser.add_argument("--tau", type=float, default=0.005,
                       help="Soft update parameter for target networks")
    
    # Other parameters
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
    print("QMIX TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Total Episodes: {args.total_episodes}")
    print(f"Max Steps per Episode: {args.max_steps_per_ep}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Dimensions: {args.hidden_dims}")
    print(f"Mixing Embed Dim: {args.mixing_embed_dim}")
    print(f"Hypernet Embed Dim: {args.hypernet_embed_dim}")
    print(f"State Dimension: {args.state_dim}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} ({args.epsilon_decay_episodes} episodes)")
    print(f"Train Frequency: {args.train_frequency}")
    print(f"Target Update Interval: {args.target_update_interval}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Run training
    train_qmix(
        env_id=args.env_id,
        total_episodes=args.total_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dims=args.hidden_dims,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_embed_dim=args.hypernet_embed_dim,
        state_dim=args.state_dim,
        buffer_capacity=args.buffer_capacity,
        min_buffer_size=args.min_buffer_size,
        train_frequency=args.train_frequency,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        target_update_interval=args.target_update_interval,
        grad_clip=args.grad_clip,
        tau=args.tau,
        seed=args.seed,
        device=device
    )

if __name__ == "__main__":
    main()
