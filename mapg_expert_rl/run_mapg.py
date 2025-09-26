#!/usr/bin/env python3
"""
MAPG Training Runner Script

This script provides an easy way to run MAPG training with different configurations.
You can modify the parameters below or use command-line arguments to customize training.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mapg_expert_rl.train_mapg import train_mapg

def main():
    parser = argparse.ArgumentParser(description="Run MAPG training with configurable parameters")
    
    # Environment parameters
    parser.add_argument("--env_id", type=str, default="expert_rl/GridWorld25v0", 
                       help="Environment ID")
    parser.add_argument("--max_steps_per_ep", type=int, default=200,
                       help="Maximum steps per episode")
    
    # Training parameters
    parser.add_argument("--total_episodes", type=int, default=2000,
                       help="Total number of training episodes")
    parser.add_argument("--lr", type=float, default=0.0003,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95,
                       help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    
    # PPO specific parameters
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                       help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--update_epochs", type=int, default=4,
                       help="Number of update epochs per batch")
    
    # Network parameters
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[128, 128],
                       help="Hidden layer dimensions")
    
    # Buffer parameters
    parser.add_argument("--rollout_length", type=int, default=2048,
                       help="Rollout buffer length")
    parser.add_argument("--min_buffer_size", type=int, default=1000,
                       help="Minimum buffer size before training starts")
    
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
    print("PPO TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Total Episodes: {args.total_episodes}")
    print(f"Max Steps per Episode: {args.max_steps_per_ep}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"GAE Lambda: {args.gae_lambda}")
    print(f"Clip Ratio: {args.clip_ratio}")
    print(f"Value Loss Coef: {args.value_loss_coef}")
    print(f"Entropy Coef: {args.entropy_coef}")
    print(f"Max Grad Norm: {args.max_grad_norm}")
    print(f"Update Epochs: {args.update_epochs}")
    print(f"Hidden Dimensions: {args.hidden_dims}")
    print(f"Rollout Length: {args.rollout_length}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Run training
    train_mapg(
        env_id=args.env_id,
        total_episodes=args.total_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        hidden_dims=args.hidden_dims,
        rollout_length=args.rollout_length,
        min_buffer_size=args.min_buffer_size,
        seed=args.seed,
        device=device
    )

if __name__ == "__main__":
    main()
