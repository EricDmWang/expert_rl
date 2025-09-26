# MAPG Expert RL

Multi-Agent Policy Gradient (MAPG) implementation for the Expert RL GridWorld25v0 environment, designed with stability as the primary focus.

## Overview

This package implements MAPG, a multi-agent policy gradient algorithm designed for cooperative multi-agent reinforcement learning. MAPG uses policy gradient methods with stability improvements to ensure reliable learning across multiple agents.

## Key Features

- **Stable Training**: MAPG's policy gradient approach with stability improvements
- **Generalized Advantage Estimation (GAE)**: Efficient advantage estimation with reduced variance
- **Multi-Agent Support**: Individual policy networks for each agent with shared learning
- **Comprehensive Monitoring**: Detailed training metrics and stability indicators
- **Conservative Initialization**: Small weight initialization for stable training

## Algorithm Details

### MAPG Architecture

1. **Policy Networks**: Individual actor-critic networks for each agent
2. **Clipped Objective**: Prevents destructive policy updates with ratio clipping
3. **GAE**: Computes advantages with reduced variance using λ-return
4. **Value Function**: Separate value head for each agent's policy network

### Stability Features

- **Conservative Weight Initialization**: `gain=0.01` for all layers
- **Gradient Clipping**: `max_grad_norm=0.5` to prevent exploding gradients
- **Learning Rate Scheduling**: Adaptive LR reduction based on loss
- **Advantage Normalization**: Per-agent advantage normalization
- **KL Divergence Monitoring**: Track policy changes to prevent instability

## Usage

### Basic Training

```bash
python train_mapg.py --total_episodes 1000 --max_steps_per_ep 200
```

### Advanced Configuration

```bash
python train_mapg.py \
    --total_episodes 2000 \
    --max_steps_per_ep 200 \
    --buffer_size 2048 \
    --batch_size 64 \
    --lr 3e-4 \
    --gamma 0.95 \
    --gae_lambda 0.95 \
    --clip_ratio 0.2 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.01 \
    --max_grad_norm 0.5 \
    --update_epochs 4
```

### Command Line Arguments

- `--total_episodes`: Total number of training episodes (default: 1000)
- `--max_steps_per_ep`: Maximum steps per episode (default: 200)
- `--buffer_size`: Rollout buffer size (default: 2048)
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.95)
- `--gae_lambda`: GAE lambda parameter (default: 0.95)
- `--clip_ratio`: PPO clip ratio (default: 0.2)
- `--value_loss_coef`: Value loss coefficient (default: 0.5)
- `--entropy_coef`: Entropy bonus coefficient (default: 0.01)
- `--max_grad_norm`: Maximum gradient norm (default: 0.5)
- `--update_epochs`: Update epochs per batch (default: 4)
- `--save_interval`: Model save interval (default: 100)

## Results Structure

Each training run creates a unique results directory:

```
ppo_expert_rl/results/run_i/
├── configs/
│   └── hparams.json          # Hyperparameters
├── models/
│   ├── checkpoint_ep100.pt   # Periodic checkpoints
│   ├── checkpoint_ep200.pt
│   └── final_model.pt        # Final model
├── training_log.csv          # Detailed training metrics
└── training_plots.png        # Training visualization
```

### Training Log Columns

- `episode`: Episode number
- `episode_return`: Discounted sum of rewards for the episode
- `episode_length`: Number of steps in the episode
- `policy_loss`: PPO policy loss
- `value_loss`: Value function loss
- `entropy_loss`: Entropy bonus loss
- `kl_div`: KL divergence between old and new policies
- `clipped_ratio`: Fraction of clipped policy ratios

### Training Plots

The generated plots show:
1. **Episode Returns**: Raw and moving average returns
2. **Policy Loss**: PPO clipped policy loss over updates
3. **KL Divergence**: Policy change monitoring
4. **Clipped Ratio**: Fraction of clipped updates

## Environment Compatibility

Designed specifically for the `expert_rl/GridWorld25v0` environment:
- 25x25 grid world with 4 agents
- 4 food items to collect cooperatively
- 18-dimensional observations per agent
- 6 actions per agent (no-op, up, down, left, right, collect)

## Stability Mechanisms

### 1. **Clipped Objective**
```python
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### 2. **Conservative Initialization**
```python
nn.init.orthogonal_(module.weight, gain=0.01)  # Very small gain
```

### 3. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm=0.5)
```

### 4. **Advantage Normalization**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 5. **Learning Rate Scheduling**
```python
scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.8)
```

## Key Advantages

1. **Proven Stability**: PPO is renowned for stable training across diverse environments
2. **Sample Efficiency**: On-policy learning with efficient advantage estimation
3. **Robust Hyperparameters**: Works well with default settings
4. **Multi-Agent Ready**: Individual networks with shared learning principles
5. **Comprehensive Monitoring**: Detailed stability and performance metrics

## Comparison with Value-Based Methods

| Feature | PPO | QMIX | MADQN |
|---------|-----|------|-------|
| Training Stability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Sample Efficiency | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hyperparameter Sensitivity | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Multi-Agent Coordination | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Implementation Complexity | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Pandas
- tqdm
- expert_rl package

## Stability Best Practices

1. **Monitor KL Divergence**: Should stay below 0.1 for stable training
2. **Watch Clipped Ratio**: High clipping indicates learning instability
3. **Track Learning Rates**: Should decrease over time for convergence
4. **Check Advantage Variance**: Normalized advantages should have unit variance

## Citation

Based on the PPO algorithm from:
```
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```

## Troubleshooting

### Common Issues and Solutions

1. **High KL Divergence**: Reduce learning rate or increase clip ratio
2. **Low Clipped Ratio**: Increase clip ratio or reduce learning rate
3. **Unstable Returns**: Check advantage normalization and gradient clipping
4. **Slow Learning**: Increase learning rate or reduce update epochs

The PPO implementation prioritizes stability over raw performance, making it ideal for reliable training runs and research applications.
