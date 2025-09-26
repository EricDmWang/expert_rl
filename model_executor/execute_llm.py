#!/usr/bin/env python3
"""
Execute LBF SmallTabNet Expert Policy (lbf_small.pt) on GridWorld25v0

- Loads lbf_llm/lbf_small.pt (weights + mu/std)
- Uses the same model for all four agents
- Executes multiple runs with a fixed seed (configurable)
- Saves CSV, JSON summary, plots, and renders (PNGs + GIFs) to
  model_executor/results/llm/execution_i
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expert_rl import GridWorld25v0


class GaussianNoise(nn.Module):
	def __init__(self, sigma=0.05):
		super().__init__(); self.sigma = float(sigma)
	def forward(self, x):
		if self.training and self.sigma > 0:
			return x + torch.randn_like(x) * self.sigma
		return x


class SmallTabNet(nn.Module):
	def __init__(self, in_dim=18, hidden1=64, hidden2=64, out_dim=6, p_drop=0.1, noise=0.05):
		super().__init__()
		self.noise = GaussianNoise(noise)
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, hidden1),
			nn.LayerNorm(hidden1),
			nn.SiLU(),
			nn.Dropout(p_drop),
			nn.Linear(hidden1, hidden2),
			nn.SiLU(),
			nn.Dropout(p_drop),
			nn.Linear(hidden2, out_dim),
		)
	def forward(self, x):
		param_device = next(self.parameters()).device
		if x.device != param_device:
			x = x.to(param_device)
		x = self.noise(x)
		return self.mlp(x)


def load_lbf_model(ckpt_path: Path, device: torch.device):
	ckpt = torch.load(str(ckpt_path), map_location=device)
	model = SmallTabNet(in_dim=18, hidden1=64, hidden2=64, out_dim=6, p_drop=0.0, noise=0.0).to(device)
	state = ckpt.get("state_dict", ckpt)
	model.load_state_dict(state)
	model.eval()
	mu = ckpt["mu"].to(device=device, dtype=torch.float32)
	std = ckpt["std"].to(device=device, dtype=torch.float32)
	return model, mu, std


def execute_single_episode(model: nn.Module, mu: torch.Tensor, std: torch.Tensor,
							 device: torch.device, gamma: float,
							 max_steps: int = 200, render: bool = False) -> Dict[str, Any]:
	# Create environment fresh for each run (mode_2 as specified)
	env = GridWorld25v0(max_steps=max_steps, gamma=gamma, mode="mode_2")
	obs, info = env.reset()
	returns_per_agent: List[List[float]] = [[] for _ in range(4)]
	length = 0
	terminated = False
	truncated = False
	
	# Optional rendering directory will be set by caller (via env._render_out_dir)
	
	while not (terminated or truncated) and length < max_steps:
		# Compute actions for all agents using the same model
		actions = []
		for i in range(4):
			obs_i = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
			obs_i = (obs_i - mu) / std
			with torch.no_grad():
				logits = model(obs_i)
				action = logits.argmax(dim=1).item()
			actions.append(action)
		
		next_obs, rewards, terminated, truncated, _ = env.step(actions)
		for i in range(4):
			returns_per_agent[i].append(rewards[i])
		
		if render:
			try:
				env.render()
			except Exception:
				pass
		
		obs = next_obs
		length += 1
	
	# Discounted episodic return per agent
	agent_returns = []
	for i in range(4):
		agent_returns.append(sum((gamma ** t) * r for t, r in enumerate(returns_per_agent[i])))
	
	# Save animation if rendering
	if render:
		try:
			env.save_animation()
		except Exception:
			pass
	
	env.close()
	return {
		"episode_return": float(sum(agent_returns)),
		"episode_length": int(length),
		"agent_returns": [float(x) for x in agent_returns],
		"total_rewards": [float(sum(rs)) for rs in returns_per_agent],
		"terminated": bool(terminated),
		"truncated": bool(truncated)
	}


def main():
	parser = argparse.ArgumentParser(description="Execute LBF SmallTabNet expert policy on GridWorld25v0")
	parser.add_argument("--lbf_model_path", type=str, default="lbf_llm/lbf_small.pt", help="Path to lbf_small.pt")
	parser.add_argument("--num_runs", type=int, default=5, help="Number of runs")
	parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
	parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
	parser.add_argument("--seed", type=int, default=1, help="Random seed (fixed for all runs)")
	parser.add_argument("--render_first_n", type=int, default=3, help="Render first N runs")
	args = parser.parse_args()
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	# Set seeds (fixed for all runs)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	# Load model
	model_path = Path(args.lbf_model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"LBF model not found: {model_path}")
	model, mu, std = load_lbf_model(model_path, device)
	
	# Prepare results directories
	results_root = Path("model_executor/results/llm")
	results_root.mkdir(parents=True, exist_ok=True)
	existing = list(results_root.glob("execution_*"))
	exec_id = len(existing)
	exec_dir = results_root / f"execution_{exec_id}"
	exec_dir.mkdir(parents=True, exist_ok=True)
	renders_dir = exec_dir / "renders"
	renders_dir.mkdir(exist_ok=True)
	print(f"Results will be saved to: {exec_dir}")
	
	# Run executions
	results: List[Dict[str, Any]] = []
	pbar = tqdm(range(args.num_runs), desc="Executing LLM Expert")
	for run_idx in pbar:
		# Create env per run via executor function; set render dir if needed
		should_render = run_idx < args.render_first_n
		if should_render:
			run_render_dir = renders_dir / f"run_{run_idx}"
			run_render_dir.mkdir(exist_ok=True)
			# The env is created inside execute_single_episode, so we pass by setting global after creation.
			# We'll set a temporary attribute used by GridWorld25v0 after first render call; to ensure, we set env attribute from inside function.
			# To handle that, we'll temporarily set an environment variable the env can read (not needed if env supports attribute assignment externally).
		
		# Execute episode
		# We call execute and then move/copy generated GIF if exists
		result = execute_single_episode(model, mu, std, device, args.gamma, args.max_steps, render=should_render)
		
		# Attach run metadata
		result.update({
			"run": run_idx,
			"seed": args.seed,
			"model_path": str(model_path)
		})
		results.append(result)
		
		pbar.set_postfix({
			"Run": run_idx,
			"Return": f"{result['episode_return']:.3f}",
			"Length": result['episode_length']
		})
	
	# Save CSV
	import pandas as pd
	df = pd.DataFrame(results)
	df.to_csv(exec_dir / "execution_results.csv", index=False)
	
	# Save summary JSON
	execution_info = {
		"execution_metadata": {
			"execution_id": exec_id,
			"timestamp": datetime.now().isoformat(),
			"device": str(device),
			"num_runs": int(args.num_runs),
			"max_steps_per_run": int(args.max_steps),
			"render_first_n_runs": int(args.render_first_n),
			"training_seed": int(args.seed),
			"seeds_used": [int(args.seed)] * int(args.num_runs)
		},
		"model_information": {
			"model_path": str(model_path),
			"model_type": "llm_expert",
			"algorithm_name": "llm",
			"hyperparameters": {
				"gamma": float(args.gamma)
			}
		},
		"environment_information": {
			"env_name": "expert_rl/GridWorld25v0",
			"env_class": "GridWorld25v0",
			"num_agents": 4,
			"obs_dim": 18,
			"num_actions": 6,
			"gamma": float(args.gamma),
			"max_steps": int(args.max_steps)
		},
		"execution_results": {
			"mean_episode_return": float(df['episode_return'].mean()),
			"std_episode_return": float(df['episode_return'].std()),
			"min_episode_return": float(df['episode_return'].min()),
			"max_episode_return": float(df['episode_return'].max()),
			"mean_episode_length": float(df['episode_length'].mean()),
			"std_episode_length": float(df['episode_length'].std()),
			"terminated_rate": float(df['terminated'].mean()),
			"truncated_rate": float(df['truncated'].mean())
		}
	}
	with open(exec_dir / "execution_info.json", 'w') as f:
		json.dump(execution_info, f, indent=2)
	
	# Simple plots
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].plot(df['episode_return'], alpha=0.8)
	axes[0].axhline(df['episode_return'].mean(), color='red', linestyle='--', label=f"Mean {df['episode_return'].mean():.3f}")
	axes[0].set_title('Episode Returns')
	axes[0].set_xlabel('Run')
	axes[0].set_ylabel('Return')
	axes[0].legend(); axes[0].grid(True, alpha=0.3)
	axes[1].plot(df['episode_length'], alpha=0.8)
	axes[1].axhline(df['episode_length'].mean(), color='red', linestyle='--', label=f"Mean {df['episode_length'].mean():.1f}")
	axes[1].set_title('Episode Lengths')
	axes[1].set_xlabel('Run')
	axes[1].set_ylabel('Length')
	axes[1].legend(); axes[1].grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(exec_dir / "execution_plots.png", dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"\nLLM expert execution completed. Results saved to: {exec_dir}")


if __name__ == "__main__":
	main()
