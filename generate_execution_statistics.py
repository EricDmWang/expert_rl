#!/usr/bin/env python3
"""
Generate execution statistics from CSV data.
This script analyzes episode_return data grouped by algorithm and produces
detailed statistical analysis and summary tables.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

def calculate_statistics(data):
    """Calculate comprehensive statistics for a dataset."""
    if len(data) == 0:
        return None
    
    # Basic statistics
    count = len(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data, ddof=1)  # Sample standard deviation
    cv = std_val / mean_val if mean_val != 0 else 0
    
    # Range statistics
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    q1 = np.percentile(data, 25)  # First quartile (Q1)
    q3 = np.percentile(data, 75)  # Third quartile (Q3)
    q25 = q1  # Alias for consistency
    q75 = q3  # Alias for consistency
    iqr = q3 - q1  # Interquartile range
    
    # Distribution shape
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    mad = np.median(np.abs(data - median_val))
    
    return {
        'count': count,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'cv': cv,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'q1': q1,
        'q3': q3,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mad': mad
    }

def interpret_distribution(stats_dict):
    """Generate interpretation of distribution characteristics."""
    skewness = stats_dict['skewness']
    kurtosis = stats_dict['kurtosis']
    cv = stats_dict['cv']
    
    # Skewness interpretation
    if abs(skewness) < 0.5:
        skew_desc = "approximately symmetric"
    elif skewness > 0.5:
        skew_desc = "positively skewed (right-tailed)"
    else:
        skew_desc = "negatively skewed (left-tailed)"
    
    # Kurtosis interpretation
    if kurtosis > 0.5:
        kurt_desc = "heavy tails (leptokurtic)"
    elif kurtosis < -0.5:
        kurt_desc = "light tails (platykurtic)"
    else:
        kurt_desc = "approximately normal tails (mesokurtic)"
    
    # Variability interpretation
    if cv < 0.2:
        var_desc = "low variability relative to mean"
    elif cv < 0.5:
        var_desc = "moderate variability relative to mean"
    else:
        var_desc = "high variability relative to mean"
    
    return {
        'skewness_desc': skew_desc,
        'kurtosis_desc': kurt_desc,
        'variability_desc': var_desc
    }

def get_algorithm_display_name(algorithm):
    """Convert algorithm names to display format."""
    name_mapping = {
        'llm_policy': 'LLM',
        'mapg_llm_expert': 'LLM Expert',
        'mapg_no_expert': 'No Expert',
        'mapg_ep250': 'EP250',
        'mapg_ep500': 'EP500',
        'mapg_ep1500': 'EP1500'
    }
    return name_mapping.get(algorithm, algorithm)

def generate_detailed_statistics(df, output_file):
    """Generate detailed statistics file."""
    algorithms = df['algorithm'].unique()
    
    with open(output_file, 'w') as f:
        f.write("EXECUTION RESULTS - STATISTICAL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        for algorithm in algorithms:
            algorithm_data = df[df['algorithm'] == algorithm]['episode_return'].values
            stats_dict = calculate_statistics(algorithm_data)
            interpretation = interpret_distribution(stats_dict)
            
            display_name = get_algorithm_display_name(algorithm)
            
            f.write(f"ALGORITHM: {display_name.upper()}\n")
            f.write("-" * 40 + "\n")
            
            # Basic Statistics
            f.write("Basic Statistics:\n")
            f.write(f"  Count:                    {stats_dict['count']:4d}\n")
            f.write(f"  Mean:                   {stats_dict['mean']:8.4f}\n")
            f.write(f"  Median:                 {stats_dict['median']:8.4f}\n")
            f.write(f"  Standard Deviation:     {stats_dict['std']:8.4f}\n")
            f.write(f"  Coefficient of Var:     {stats_dict['cv']:8.4f}\n\n")
            
            # Range Statistics
            f.write("Range Statistics:\n")
            f.write(f"  Minimum:               {stats_dict['min']:8.4f}\n")
            f.write(f"  Maximum:               {stats_dict['max']:8.4f}\n")
            f.write(f"  Range:                 {stats_dict['range']:8.4f}\n")
            f.write(f"  Q1 (25th percentile):    {stats_dict['q1']:8.4f}\n")
            f.write(f"  Q3 (75th percentile):    {stats_dict['q3']:8.4f}\n")
            f.write(f"  Q25 (25th percentile):   {stats_dict['q25']:8.4f}\n")
            f.write(f"  Q75 (75th percentile):   {stats_dict['q75']:8.4f}\n")
            f.write(f"  IQR (Interquartile):   {stats_dict['iqr']:8.4f}\n\n")
            
            # Distribution Shape
            f.write("Distribution Shape:\n")
            f.write(f"  Skewness:              {stats_dict['skewness']:8.4f}\n")
            f.write(f"  Kurtosis:              {stats_dict['kurtosis']:8.4f}\n")
            f.write(f"  Median Abs Dev:        {stats_dict['mad']:8.4f}\n\n")
            
            # Interpretation
            f.write("Interpretation:\n")
            f.write(f"  - Distribution is {interpretation['skewness_desc']}\n")
            f.write(f"  - Distribution has {interpretation['kurtosis_desc']}\n")
            f.write(f"  - {interpretation['variability_desc']}\n\n")
            
            f.write("=" * 60 + "\n\n")
        
        # Comparative Analysis
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        # Performance ranking
        algorithm_stats = []
        for algorithm in algorithms:
            algorithm_data = df[df['algorithm'] == algorithm]['episode_return'].values
            stats_dict = calculate_statistics(algorithm_data)
            display_name = get_algorithm_display_name(algorithm)
            algorithm_stats.append((display_name, stats_dict))
        
        # Sort by mean performance (descending)
        algorithm_stats.sort(key=lambda x: x[1]['mean'], reverse=True)
        
        f.write("Performance Ranking (by Mean):\n")
        for i, (display_name, stats_dict) in enumerate(algorithm_stats, 1):
            f.write(f"  {i}. {display_name.lower():<15} Mean:   {stats_dict['mean']:8.4f} ± {stats_dict['std']:8.4f}\n")
        
        f.write("\nConsistency Ranking (by Coefficient of Variation, lower is better):\n")
        # Sort by coefficient of variation (ascending - lower is better)
        algorithm_stats.sort(key=lambda x: x[1]['cv'])
        for i, (display_name, stats_dict) in enumerate(algorithm_stats, 1):
            f.write(f"  {i}. {display_name.lower():<15} CV:   {stats_dict['cv']:8.4f}\n")
        
        f.write("\nBest Performer (highest mean):\n")
        best_algo = max(algorithm_stats, key=lambda x: x[1]['mean'])
        f.write(f"  Algorithm: {best_algo[0].lower()}\n")
        f.write(f"  Mean: {best_algo[1]['mean']:.4f}\n\n")
        
        f.write("Worst Performer (lowest mean):\n")
        worst_algo = min(algorithm_stats, key=lambda x: x[1]['mean'])
        f.write(f"  Algorithm: {worst_algo[0].lower()}\n")
        f.write(f"  Mean: {worst_algo[1]['mean']:.4f}\n\n")

def generate_summary_table(df, output_file):
    """Generate summary table file."""
    algorithms = df['algorithm'].unique()
    
    with open(output_file, 'w') as f:
        f.write("EXECUTION RESULTS - SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        
        # Table header
        f.write(f"{'Algorithm':<15} {'Count':<6} {'Mean':<8} {'Median':<8} {'Q1':<8} {'Q3':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'CV':<8}\n")
        f.write("-" * 100 + "\n")
        
        # Table rows
        for algorithm in algorithms:
            algorithm_data = df[df['algorithm'] == algorithm]['episode_return'].values
            stats_dict = calculate_statistics(algorithm_data)
            display_name = get_algorithm_display_name(algorithm)
            
            f.write(f"{display_name:<15} {stats_dict['count']:<6} {stats_dict['mean']:<8.4f} "
                   f"{stats_dict['median']:<8.4f} {stats_dict['q1']:<8.4f} {stats_dict['q3']:<8.4f} "
                   f"{stats_dict['std']:<8.4f} {stats_dict['min']:<8.4f} {stats_dict['max']:<8.4f} "
                   f"{stats_dict['cv']:<8.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("Legend:\n")
        f.write("  Count: Number of episodes\n")
        f.write("  Mean: Average episode return\n")
        f.write("  Median: Middle value (50th percentile)\n")
        f.write("  Q1: First quartile (25th percentile)\n")
        f.write("  Q3: Third quartile (75th percentile)\n")
        f.write("  Std: Standard deviation\n")
        f.write("  Min/Max: Minimum and maximum values\n")
        f.write("  CV: Coefficient of variation (std/mean)\n")

def main():
    """Main function to generate statistics from CSV data."""
    # File paths
    csv_file = "/home/dongmingwang/project/Expert_RL/experiment_results/execution/all_policies_execution_results.csv"
    stats_output = "/home/dongmingwang/project/Expert_RL/experiment_results/execution/execution_statistics_generated.txt"
    summary_output = "/home/dongmingwang/project/Expert_RL/experiment_results/execution/execution_summary_table_generated.txt"
    
    # Read CSV data
    print("Reading CSV data...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Algorithms found: {df['algorithm'].unique()}")
    print(f"Unique policies: {df['policy_name'].unique()}")
    
    # Generate detailed statistics
    print("Generating detailed statistics...")
    generate_detailed_statistics(df, stats_output)
    print(f"Detailed statistics written to: {stats_output}")
    
    # Generate summary table
    print("Generating summary table...")
    generate_summary_table(df, summary_output)
    print(f"Summary table written to: {summary_output}")
    
    print("Statistics generation completed!")

if __name__ == "__main__":
    main()
