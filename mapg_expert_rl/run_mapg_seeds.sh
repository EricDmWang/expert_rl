#!/bin/bash

# MAPG Training Script with Multiple Random Seeds
# This script runs MAPG training with different random seeds for reproducibility testing

# Activate conda environment
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MAPG Training with Multiple Seeds..."
echo "=============================================="

# Configuration parameters
TOTAL_EPISODES=2000
MAX_STEPS_PER_EP=200
SEED_START=1
SEED_END=5

# Run training with different seeds
for SEED in $(seq $SEED_START $SEED_END); do
    echo ""
    echo "Running MAPG with seed $SEED ($((SEED - SEED_START + 1))/$((SEED_END - SEED_START + 1)))..."
    echo "----------------------------------------"
    
    # Run training with current seed
    python mapg_expert_rl/train_mapg.py \
        --total_episodes $TOTAL_EPISODES \
        --max_steps_per_ep $MAX_STEPS_PER_EP \
        --seed $SEED \
        --lr 3e-4 \
        --gamma 0.97 \
        --buffer_size 2048 \
        --batch_size 64 \
        --clip_ratio 0.2
    
    EXIT_CODE=$?
    
    # Check if training completed (even with segfault, if models were saved)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed successfully for seed $SEED"
    elif [ $EXIT_CODE -eq 139 ]; then
        # Segmentation fault (139), but check if models were saved
        LATEST_RUN=$(ls -td mapg_expert_rl/results/run_* 2>/dev/null | head -1)
        if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/models" ] && [ $(ls -1 "$LATEST_RUN/models"/*.pt 2>/dev/null | wc -l) -gt 0 ]; then
            echo "⚠ Training completed with segfault but models saved for seed $SEED"
        else
            echo "✗ Training failed with segfault for seed $SEED"
            exit 1
        fi
    else
        echo "✗ Training failed for seed $SEED (exit code: $EXIT_CODE)"
        exit 1
    fi
    
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
done

echo ""
echo "=============================================="
echo "All training runs completed successfully!"
echo "Results saved in: mapg_expert_rl/results/run_*"
echo "=============================================="
