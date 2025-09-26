#!/bin/bash

# QMIX Training Script
# This script provides examples of how to run QMIX with different configurations

# Activate conda environment (adjust if your environment name is different)
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting QMIX Training..."

# Example 1: Default configuration
echo "Running with default configuration..."
python qmix_expert_rl/train_qmix.py

# Example 2: Quick training run (fewer episodes)
# echo "Running quick training (500 episodes)..."
# python qmix_expert_rl/train_qmix.py --total_episodes 500 --epsilon_decay_episodes 200

# Example 3: Conservative QMIX configuration
# echo "Running conservative QMIX..."
# python qmix_expert_rl/train_qmix.py --lr 0.00005 --epsilon_decay_episodes 800 --total_episodes 3000

# Example 4: Aggressive QMIX configuration
# echo "Running aggressive QMIX..."
# python qmix_expert_rl/train_qmix.py --lr 0.0005 --epsilon_decay_episodes 300 --total_episodes 1500

# Example 5: Large network configuration
# echo "Running with larger network..."
# python qmix_expert_rl/train_qmix.py --hidden_dims 256 256 128 --mixing_embed_dim 128 --total_episodes 2500

# Example 6: Fast exploration
# echo "Running with fast exploration..."
# python qmix_expert_rl/train_qmix.py --epsilon_decay_episodes 200 --total_episodes 1500

# Example 7: Large mixing network
# echo "Running with large mixing network..."
# python qmix_expert_rl/train_qmix.py --mixing_embed_dim 128 --hypernet_embed_dim 256 --total_episodes 2500

# Example 8: High training frequency
# echo "Running with high training frequency..."
# python qmix_expert_rl/train_qmix.py --train_frequency 4 --total_episodes 2000

echo "Training completed!"
