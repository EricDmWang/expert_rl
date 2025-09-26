#!/bin/bash

# MAPG Training with LBF Expert Guidance
# =====================================

# Set default parameters
TOTAL_EPISODES=2000
MAX_STEPS_PER_EP=200
BUFFER_SIZE=2048
BATCH_SIZE=128
LR=3e-4
GAMMA=0.97
GAE_LAMBDA=0.95
CLIP_RATIO=0.2
VALUE_LOSS_COEF=0.5
ENTROPY_COEF=0.01
MAX_GRAD_NORM=0.5
UPDATE_EPOCHS=4
SAVE_INTERVAL=100
SEED=1
MODE="mode_2"

# Expert guidance parameters
EXPERT_GUIDANCE_COEF=0.05
EXPERT_BETA_0=0.1
LBF_MODEL_PATH="lbf_llm/lbf_small.pt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --total_episodes)
            TOTAL_EPISODES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --expert_guidance_coef)
            EXPERT_GUIDANCE_COEF="$2"
            shift 2
            ;;
        --expert_beta_0)
            EXPERT_BETA_0="$2"
            shift 2
            ;;
        --lbf_model_path)
            LBF_MODEL_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "Starting MAPG Training with LBF Expert Guidance..."
echo "=================================================="
echo "Configuration:"
echo "  Total Episodes: $TOTAL_EPISODES"
echo "  Seed: $SEED"
echo "  Mode: $MODE"
echo "  Learning Rate: $LR"
echo "  Expert Guidance Coef: $EXPERT_GUIDANCE_COEF"
echo "  Expert Beta 0: $EXPERT_BETA_0"
echo "  LBF Model Path: $LBF_MODEL_PATH"
echo ""

# Run training
python mapg_expert_rl/train_mapg_llm.py \
    --total_episodes $TOTAL_EPISODES \
    --max_steps_per_ep $MAX_STEPS_PER_EP \
    --buffer_size $BUFFER_SIZE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --clip_ratio $CLIP_RATIO \
    --value_loss_coef $VALUE_LOSS_COEF \
    --entropy_coef $ENTROPY_COEF \
    --max_grad_norm $MAX_GRAD_NORM \
    --update_epochs $UPDATE_EPOCHS \
    --save_interval $SAVE_INTERVAL \
    --seed $SEED \
    --mode $MODE \
    --expert_guidance_coef $EXPERT_GUIDANCE_COEF \
    --expert_beta_0 $EXPERT_BETA_0 \
    --lbf_model_path $LBF_MODEL_PATH

echo ""
echo "Training completed!"
