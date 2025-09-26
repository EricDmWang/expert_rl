#!/usr/bin/env python3
"""
Example Usage Script for Model Executor

This script demonstrates how to use the model executor to evaluate trained models
from different algorithms.
"""

import subprocess
import sys
from pathlib import Path

def run_execution(model_path: str, num_runs: int = 30, max_steps: int = 100, render_first_n: int = 3):
    """Run model execution with specified parameters."""
    
    cmd = [
        sys.executable, "model_executor/execute_models.py",
        model_path,
        "--num_runs", str(num_runs),
        "--max_steps", str(max_steps),
        "--render_first_n", str(render_first_n)
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("Execution completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("Execution failed!")
            print("\nError:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Exception during execution: {e}")
        return False

def main():
    """Main function demonstrating different usage patterns."""
    
    print("Model Executor Example Usage")
    print("=" * 60)
    
    # Example 1: Quick evaluation of MADQN model
    print("\nExample 1: Quick MADQN Evaluation")
    print("Running 20 episodes with 50 steps each, rendering first 2 episodes...")
    
    success = run_execution(
        model_path="madqn/results/run_13",
        num_runs=30,
        max_steps=200,
        render_first_n=2
    )
    
    if success:
        print("MADQN evaluation completed!")
    else:
        print("MADQN evaluation failed!")
    
    # Example 2: Comprehensive QMIX evaluation
    print("\nExample 2: Comprehensive QMIX Evaluation")
    print("Running 30 episodes with 100 steps each, rendering first 5 episodes...")
    
    success = run_execution(
        model_path="qmix_expert_rl/results/run_11",
        num_runs=30,
        max_steps=100,
        render_first_n=5
    )
    
    if success:
        print("QMIX evaluation completed!")
    else:
        print("QMIX evaluation failed!")
    
    # Example 3: MAPG evaluation (if available)
    print("\nExample 3: MAPG Evaluation")
    print("Running 30 episodes with 100 steps each, rendering first 3 episodes...")
    
    # Check if MAPG results exist
    mapg_path = "mapg_expert_rl/results"
    if Path(mapg_path).exists():
        mapg_runs = list(Path(mapg_path).glob("run_*"))
        if mapg_runs:
            latest_run = sorted(mapg_runs)[-1]
            success = run_execution(
                model_path=str(latest_run),
                num_runs=30,
                max_steps=100,
                render_first_n=3
            )
            
            if success:
                print("MAPG evaluation completed!")
            else:
                print("MAPG evaluation failed!")
        else:
            print("No MAPG runs found, skipping...")
    else:
        print("MAPG results directory not found, skipping...")
    
    print("\nExample execution completed!")
    print("\nCheck the following directories for results:")
    print("   - model_executor/results/")
    print("   - Each execution creates a new subdirectory with CSV, JSON, and PNG files")
    print("   - Rendered episodes are saved in the 'renders/' subdirectory")

if __name__ == "__main__":
    main()
