# QMIX Expert RL

QMIX implementation for the Expert RL GridWorld25v0 environment, based on PyMARL2's proven QMIX algorithm.

## Overview

This package implements QMIX (Q-learning with Monotonic Value Function Factorisation) for cooperative multi-agent reinforcement learning on the custom GridWorld25v0 environment. QMIX uses value decomposition to learn a monotonic mixing function that combines individual agent Q-values into a global Q-value.

## Key Features

- **Value Decomposition**: Monotonic mixing network that ensures individual agent improvements lead to global improvements
- **Stable Training**: Includes proven tricks from PyMARL2 for stable multi-agent learning
- **Comprehensive Logging**: Detailed training metrics, model checkpoints, and visualization
- **Flexible Configuration**: Configurable hyperparameters and network architectures

## Algorithm Details

### QMIX Architecture

1. **Individual Q-Networks**: Each agent has its own Q-network for action selection
2. **Mixing Network**: Hypernetwork-based mixer that combines individual Q-values into global Q-value
3. **Monotonicity Constraint**: Ensures positive weights in the mixing network
4. **Target Networks**: Separate target networks for stable Q-learning

### Training Features

- **Experience Replay**: Large replay buffer for sample efficiency
- **Target Network Updates**: Periodic hard updates of target networks
- **Epsilon-Greedy Exploration**: Decaying exploration strategy
- **Gradient Clipping**: Prevents exploding gradients
- **Orthogonal Initialization**: Stable network initialization

## Usage

### Basic Training

```bash
python train_qmix.py --total_episodes 1000 --max_steps_per_ep 200
```

### Advanced Configuration

```bash
python train_qmix.py \
    --total_episodes 2000 \
    --max_steps_per_ep 200 \
    --buffer_capacity 100000 \
    --batch_size 64 \
    --lr 1e-3 \
    --gamma 0.95 \
    --target_update_interval 200 \
    --train_frequency 4 \
    --min_buffer_size 10000 \
    --save_interval 100
```

### Command Line Arguments

- `--total_episodes`: Total number of training episodes (default: 1000)
- `--max_steps_per_ep`: Maximum steps per episode (default: 200)
- `--buffer_capacity`: Replay buffer capacity (default: 100000)
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--gamma`: Discount factor (default: 0.95)
- `--target_update_interval`: Target network update interval (default: 200)
- `--train_frequency`: Training frequency (default: 4)
- `--min_buffer_size`: Minimum buffer size before training (default: 10000)
- `--save_interval`: Model save interval (default: 100)

## Results Structure

Each training run creates a unique results directory:

```
qmix_expert_rl/results/run_i/
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
- `epsilon`: Current exploration rate
- `loss`: QMIX training loss
- `td_error`: Temporal difference error
- `q_value`: Average Q-value

### Training Plots

The generated plots show:
1. **Episode Returns**: Raw and moving average returns
2. **Training Loss**: QMIX loss over training steps
3. **Q-Values**: Average Q-value evolution
4. **Epsilon and TD Error**: Exploration and learning metrics

## Environment Compatibility

Designed specifically for the `expert_rl/GridWorld25v0` environment:
- 25x25 grid world with 4 agents
- 4 food items to collect cooperatively
- 18-dimensional observations per agent
- 6 actions per agent (no-op, up, down, left, right, collect)

## Key Advantages

1. **Proven Performance**: Based on PyMARL2's state-of-the-art QMIX implementation
2. **Cooperative Learning**: Value decomposition ensures coordinated behavior
3. **Sample Efficiency**: Experience replay and target networks for stable learning
4. **Comprehensive Monitoring**: Detailed logging and visualization for analysis
5. **Easy Integration**: Compatible with the expert_rl environment structure

## Implementation Details

### Network Architecture

- **Q-Networks**: 2-layer MLPs with LayerNorm and ReLU activations
- **Mixing Network**: Hypernetwork with 2-layer MLPs for weight generation
- **Embedding Dimensions**: Configurable mixing and hypernetwork embedding sizes

### Training Tricks

- Orthogonal weight initialization
- Gradient clipping (max norm: 10.0)
- Adam optimizer with weight decay
- Epsilon-greedy exploration with linear decay
- Target network updates every 200 steps

## Comparison with MADQN

| Feature | QMIX | MADQN |
|---------|------|-------|
| Value Decomposition | Monotonic mixing | Independent Q-learning |
| Global Coordination | Yes (via mixer) | Limited (centralized critic) |
| Training Stability | High (proven tricks) | Good (gradient clipping) |
| Sample Efficiency | High | Medium |
| Implementation Complexity | Medium | High |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Pandas
- tqdm
- expert_rl package

## Citation

Based on the QMIX algorithm from:
```
@article{rashid2018qmix,
  title={Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning},
  author={Rashid, Tabish and Samvelyan, Mikayel and Schroeder, Christian and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  journal={International conference on machine learning},
  pages={4295--4304},
  year={2018},
  organization={PMLR}
}
```

And implementation tricks from PyMARL2:
```
@article{hu2021rethinking,
  title={Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning},
  author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
  year={2021},
  eprint={2102.03479},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
