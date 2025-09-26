#!/bin/bash

# MAPG Training Script
# This script provides examples of how to run MAPG with different configurations

# Activate conda environment (adjust if your environment name is different)
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MAPG Training..."

# Example 1: Default configuration
echo "Running with default configuration..."
python mapg_expert_rl/train_mapg.py

# Example 2: Quick training run (fewer episodes)
# echo "Running quick training (500 episodes)..."
# python mapg_expert_rl/train_mapg.py --total_episodes 500 --rollout_length 1024

# Example 3: Conservative PPO configuration
# echo "Running conservative PPO..."
# python mapg_expert_rl/train_mapg.py --lr 0.0001 --clip_ratio 0.1 --total_episodes 3000

# Example 4: Aggressive PPO configuration
# echo "Running aggressive PPO..."
# python mapg_expert_rl/train_mapg.py --lr 0.001 --clip_ratio 0.3 --total_episodes 1500

# Example 5: Large network configuration
# echo "Running with larger network..."
# python mapg_expert_rl/train_mapg.py --hidden_dims 256 256 128 --total_episodes 2500

# Example 6: High entropy exploration
# echo "Running with high entropy..."
# python mapg_expert_rl/train_mapg.py --entropy_coef 0.05 --total_episodes 2000

# Example 7: Focused on value learning
# echo "Running with high value loss coefficient..."
# python mapg_expert_rl/train_mapg.py --value_loss_coef 1.0 --total_episodes 2000

echo "Training completed!"
echo ""
echo "To run with multiple random seeds (1-30), use:"
echo "  ./mapg_expert_rl/run_mapg_seeds.sh"
