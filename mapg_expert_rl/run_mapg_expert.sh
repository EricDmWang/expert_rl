#!/bin/bash

# MAPG with Expert Guidance Training Script
# This script runs MAPG training with expert guidance from MADQN models

# Activate conda environment
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MAPG with Expert Guidance Training..."
echo "=============================================="

# Configuration parameters
TOTAL_EPISODES=2000
MAX_STEPS_PER_EP=200
SEED=1  # Same seed as MADQN run_25
MODE="mode_2"  # Fixed mode as specified

# Expert guidance parameters (improved for better convergence)
EXPERT_GUIDANCE_COEF=0.3  # Reduced for less interference
EXPERT_BETA_0=0.5          # Reduced for gentler expert guidance
EXPERT_STAGE="all"  # Options: ep250, ep500, ep1500, all

# Expert model paths (from MADQN run_25)
EXPERT_MODEL_PATHS=(
    "madqn/results/run_25/models/checkpoint_agent_0_ep250.pt"
    "madqn/results/run_25/models/checkpoint_agent_0_ep500.pt"
    "madqn/results/run_25/models/checkpoint_agent_0_ep1500.pt"
)

echo "Using seed: $SEED (same as MADQN run_25)"
echo "Environment mode: $MODE"
echo "Expert guidance coefficient: $EXPERT_GUIDANCE_COEF"
echo "Expert beta_0: $EXPERT_BETA_0"
echo "Expert stage: $EXPERT_STAGE"
echo "Expert models:"
for path in "${EXPERT_MODEL_PATHS[@]}"; do
    echo "  - $path"
done
echo ""

if [ "$EXPERT_STAGE" = "all" ]; then
    echo "Will run training for all three expert stages: ep250, ep500, ep1500"
    echo "Each training will use a different expert policy throughout the entire training process"
    echo ""
fi

# Run training with expert guidance
python mapg_expert_rl/train_mapg_expert.py \
    --total_episodes $TOTAL_EPISODES \
    --max_steps_per_ep $MAX_STEPS_PER_EP \
    --seed $SEED \
    --mode $MODE \
    --lr 1e-5 \
    --gamma 0.97 \
    --buffer_size 2048 \
    --batch_size 64 \
    --clip_ratio 0.2 \
    --expert_guidance_coef $EXPERT_GUIDANCE_COEF \
    --expert_beta_0 $EXPERT_BETA_0 \
    --expert_stage $EXPERT_STAGE \
    --expert_model_paths "${EXPERT_MODEL_PATHS[@]}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ MAPG with Expert Guidance training completed successfully!"
    echo "Results saved in: mapg_expert_rl/results/run_*"
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "Check the results directory for:"
echo "  - Training plots with expert guidance metrics"
echo "  - Model checkpoints"
echo "  - Comprehensive training log with expert information"
echo "=============================================="
