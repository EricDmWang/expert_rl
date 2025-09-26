#!/usr/bin/env python3
"""
Simple script to generate execution statistics from CSV data.
Usage: python simple_statistics_generator.py <csv_file_path>
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
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

def generate_statistics_report(csv_file, algorithm_col='algorithm', value_col='episode_return'):
    """Generate a comprehensive statistics report from CSV data."""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    algorithms = df[algorithm_col].unique()
    
    print("EXECUTION RESULTS - STATISTICAL ANALYSIS")
    print("=" * 60)
    print()
    
    algorithm_stats = []
    
    for algorithm in algorithms:
        algorithm_data = df[df[algorithm_col] == algorithm][value_col].values
        stats_dict = calculate_statistics(algorithm_data)
        interpretation = interpret_distribution(stats_dict)
        
        print(f"ALGORITHM: {algorithm.upper()}")
        print("-" * 40)
        
        # Basic Statistics
        print("Basic Statistics:")
        print(f"  Count:                    {stats_dict['count']:4d}")
        print(f"  Mean:                   {stats_dict['mean']:8.4f}")
        print(f"  Median:                 {stats_dict['median']:8.4f}")
        print(f"  Standard Deviation:     {stats_dict['std']:8.4f}")
        print(f"  Coefficient of Var:     {stats_dict['cv']:8.4f}")
        print()
        
        # Range Statistics
        print("Range Statistics:")
        print(f"  Minimum:               {stats_dict['min']:8.4f}")
        print(f"  Maximum:               {stats_dict['max']:8.4f}")
        print(f"  Range:                 {stats_dict['range']:8.4f}")
        print(f"  Q1 (25th percentile):    {stats_dict['q1']:8.4f}")
        print(f"  Q3 (75th percentile):    {stats_dict['q3']:8.4f}")
        print(f"  Q25 (25th percentile):   {stats_dict['q25']:8.4f}")
        print(f"  Q75 (75th percentile):   {stats_dict['q75']:8.4f}")
        print(f"  IQR (Interquartile):   {stats_dict['iqr']:8.4f}")
        print()
        
        # Distribution Shape
        print("Distribution Shape:")
        print(f"  Skewness:              {stats_dict['skewness']:8.4f}")
        print(f"  Kurtosis:              {stats_dict['kurtosis']:8.4f}")
        print(f"  Median Abs Dev:        {stats_dict['mad']:8.4f}")
        print()
        
        # Interpretation
        print("Interpretation:")
        print(f"  - Distribution is {interpretation['skewness_desc']}")
        print(f"  - Distribution has {interpretation['kurtosis_desc']}")
        print(f"  - {interpretation['variability_desc']}")
        print()
        
        print("=" * 60)
        print()
        
        algorithm_stats.append((algorithm, stats_dict))
    
    # Comparative Analysis
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)
    print()
    
    # Performance ranking
    algorithm_stats.sort(key=lambda x: x[1]['mean'], reverse=True)
    
    print("Performance Ranking (by Mean):")
    for i, (algorithm, stats_dict) in enumerate(algorithm_stats, 1):
        print(f"  {i}. {algorithm:<20} Mean:   {stats_dict['mean']:8.4f} ± {stats_dict['std']:8.4f}")
    
    print()
    print("Consistency Ranking (by Coefficient of Variation, lower is better):")
    # Sort by coefficient of variation (ascending - lower is better)
    algorithm_stats.sort(key=lambda x: x[1]['cv'])
    for i, (algorithm, stats_dict) in enumerate(algorithm_stats, 1):
        print(f"  {i}. {algorithm:<20} CV:   {stats_dict['cv']:8.4f}")
    
    print()
    print("Best Performer (highest mean):")
    best_algo = max(algorithm_stats, key=lambda x: x[1]['mean'])
    print(f"  Algorithm: {best_algo[0]}")
    print(f"  Mean: {best_algo[1]['mean']:.4f}")
    print()
    
    print("Worst Performer (lowest mean):")
    worst_algo = min(algorithm_stats, key=lambda x: x[1]['mean'])
    print(f"  Algorithm: {worst_algo[0]}")
    print(f"  Mean: {worst_algo[1]['mean']:.4f}")
    print()
    
    # Summary Table
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    
    # Table header
    print(f"{'Algorithm':<20} {'Count':<6} {'Mean':<8} {'Median':<8} {'Q1':<8} {'Q3':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'CV':<8}")
    print("-" * 100)
    
    # Table rows
    for algorithm, stats_dict in algorithm_stats:
        print(f"{algorithm:<20} {stats_dict['count']:<6} {stats_dict['mean']:<8.4f} "
              f"{stats_dict['median']:<8.4f} {stats_dict['q1']:<8.4f} {stats_dict['q3']:<8.4f} "
              f"{stats_dict['std']:<8.4f} {stats_dict['min']:<8.4f} {stats_dict['max']:<8.4f} "
              f"{stats_dict['cv']:<8.4f}")
    
    print()
    print("=" * 100)
    print("Legend:")
    print("  Count: Number of episodes")
    print("  Mean: Average episode return")
    print("  Median: Middle value (50th percentile)")
    print("  Q1: First quartile (25th percentile)")
    print("  Q3: Third quartile (75th percentile)")
    print("  Std: Standard deviation")
    print("  Min/Max: Minimum and maximum values")
    print("  CV: Coefficient of variation (std/mean)")

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python simple_statistics_generator.py <csv_file_path>")
        print("Example: python simple_statistics_generator.py results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    
    try:
        generate_statistics_report(csv_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
