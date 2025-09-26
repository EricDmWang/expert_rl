#!/bin/bash

# MAPG with All Expert Stages Training Script
# This script runs MAPG training with all three expert stages (ep250, ep500, ep1500)
# Each training uses a different expert policy throughout the entire training process

# Activate conda environment
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MAPG with All Expert Stages Training..."
echo "=============================================="
echo "This will run three separate training sessions:"
echo "1. Expert Stage EP250 (Early expert policy)"
echo "2. Expert Stage EP500 (Mid-level expert policy)"
echo "3. Expert Stage EP1500 (Advanced expert policy)"
echo ""

# Configuration parameters
TOTAL_EPISODES=2000
MAX_STEPS_PER_EP=200
SEED=1  # Same seed as MADQN run_25
MODE="mode_2"  # Fixed mode as specified

# Expert guidance parameters (improved for better convergence)
EXPERT_GUIDANCE_COEF=0.1  # Reduced for less interference
EXPERT_BETA_0=0.5          # Reduced for gentler expert guidance

# Expert model paths (from MADQN run_25)
EXPERT_MODEL_PATHS=(
    "madqn/results/run_25/models/checkpoint_agent_0_ep250.pt"
    "madqn/results/run_25/models/checkpoint_agent_0_ep500.pt"
    "madqn/results/run_25/models/checkpoint_agent_0_ep1500.pt"
)

echo "Configuration:"
echo "  Total episodes per expert stage: $TOTAL_EPISODES"
echo "  Max steps per episode: $MAX_STEPS_PER_EP"
echo "  Random seed: $SEED (same as MADQN run_25)"
echo "  Environment mode: $MODE"
echo "  Expert guidance coefficient: $EXPERT_GUIDANCE_COEF"
echo "  Expert beta_0: $EXPERT_BETA_0"
echo "  Expert models:"
for path in "${EXPERT_MODEL_PATHS[@]}"; do
    echo "    - $path"
done
echo ""

# Run training for all expert stages
python mapg_expert_rl/train_mapg_expert.py \
    --total_episodes $TOTAL_EPISODES \
    --max_steps_per_ep $MAX_STEPS_PER_EP \
    --seed $SEED \
    --mode $MODE \
    --lr 3e-4 \
    --gamma 0.97 \
    --buffer_size 2048 \
    --batch_size 64 \
    --clip_ratio 0.2 \
    --expert_guidance_coef $EXPERT_GUIDANCE_COEF \
    --expert_beta_0 $EXPERT_BETA_0 \
    --expert_stage all \
    --expert_model_paths "${EXPERT_MODEL_PATHS[@]}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All MAPG Expert Stages training completed successfully!"
    echo ""
    echo "Results saved in:"
    echo "  - mapg_expert_rl/results/run_* (latest runs)"
    echo ""
    echo "Each expert stage has its own results directory:"
    echo "  - EP250: Uses early expert policy (episode 250)"
    echo "  - EP500: Uses mid-level expert policy (episode 500)"
    echo "  - EP1500: Uses advanced expert policy (episode 1500)"
    echo ""
    echo "You can compare the results to see how expert quality affects MAPG learning!"
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "Check the results directories for:"
echo "  - Training plots with expert guidance metrics"
echo "  - Model checkpoints for each expert stage"
echo "  - Comprehensive training logs with expert information"
echo "  - CSV logs with expert guidance metrics"
echo "=============================================="
