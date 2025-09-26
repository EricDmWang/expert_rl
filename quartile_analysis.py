#!/usr/bin/env python3
"""
Enhanced Quartile Analysis Script
This script provides detailed quartile analysis including Q1, Q3, and additional quartile-related statistics.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

def calculate_detailed_quartiles(data):
    """Calculate detailed quartile statistics."""
    if len(data) == 0:
        return None
    
    # Sort data for quartile calculations
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Basic quartiles using different methods
    q1_method1 = np.percentile(data, 25)  # Standard percentile method
    q2_median = np.percentile(data, 50)   # Median
    q3_method1 = np.percentile(data, 75)  # Standard percentile method
    
    # Alternative quartile calculations using different interpolation methods
    q1_linear = np.percentile(data, 25, interpolation='linear')
    q3_linear = np.percentile(data, 75, interpolation='linear')
    
    q1_midpoint = np.percentile(data, 25, interpolation='midpoint')
    q3_midpoint = np.percentile(data, 75, interpolation='midpoint')
    
    # Manual quartile calculation (R-6 method approximation)
    def manual_quartile(data, p):
        sorted_data = np.sort(data)
        n = len(sorted_data)
        if n == 0:
            return 0
        # Using R-6 method (similar to numpy's default)
        index = p * (n - 1)
        lower = int(np.floor(index))
        upper = int(np.ceil(index))
        if lower == upper:
            return sorted_data[lower]
        weight = index - lower
        return (1 - weight) * sorted_data[lower] + weight * sorted_data[upper]
    
    q1_manual = manual_quartile(data, 0.25)
    q3_manual = manual_quartile(data, 0.75)
    
    # Five-number summary
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Additional quartile-related statistics
    iqr = q3_method1 - q1_method1
    qd = iqr / 2  # Quartile deviation (semi-interquartile range)
    
    # Outlier detection using IQR method
    lower_fence = q1_method1 - 1.5 * iqr
    upper_fence = q3_method1 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    
    # Quartile skewness
    quartile_skewness = (q3_method1 + q1_method1 - 2 * q2_median) / iqr
    
    return {
        'count': n,
        'min': min_val,
        'max': max_val,
        'q1_standard': q1_method1,
        'q1_linear': q1_linear,
        'q1_midpoint': q1_midpoint,
        'q1_manual': q1_manual,
        'q2_median': q2_median,
        'q3_standard': q3_method1,
        'q3_linear': q3_linear,
        'q3_midpoint': q3_midpoint,
        'q3_manual': q3_manual,
        'iqr': iqr,
        'qd': qd,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'outlier_count': len(outliers),
        'outlier_values': outliers.tolist(),
        'quartile_skewness': quartile_skewness
    }

def generate_quartile_report(csv_file, algorithm_col='algorithm', value_col='episode_return'):
    """Generate a comprehensive quartile analysis report."""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    algorithms = df[algorithm_col].unique()
    
    print("QUARTILE ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    for algorithm in algorithms:
        algorithm_data = df[df[algorithm_col] == algorithm][value_col].values
        quartile_stats = calculate_detailed_quartiles(algorithm_data)
        
        if quartile_stats is None:
            continue
            
        print(f"ALGORITHM: {algorithm.upper()}")
        print("-" * 50)
        
        # Five-number summary
        print("Five-Number Summary:")
        print(f"  Minimum:           {quartile_stats['min']:10.4f}")
        print(f"  Q1 (First Quartile): {quartile_stats['q1_standard']:10.4f}")
        print(f"  Q2 (Median):       {quartile_stats['q2_median']:10.4f}")
        print(f"  Q3 (Third Quartile): {quartile_stats['q3_standard']:10.4f}")
        print(f"  Maximum:           {quartile_stats['max']:10.4f}")
        print()
        
        # Quartile calculation methods comparison
        print("Quartile Calculation Methods:")
        print(f"  Q1 Standard (percentile):  {quartile_stats['q1_standard']:10.4f}")
        print(f"  Q1 Linear interpolation:   {quartile_stats['q1_linear']:10.4f}")
        print(f"  Q1 Midpoint method:        {quartile_stats['q1_midpoint']:10.4f}")
        print(f"  Q1 Manual calculation:     {quartile_stats['q1_manual']:10.4f}")
        print()
        print(f"  Q3 Standard (percentile):  {quartile_stats['q3_standard']:10.4f}")
        print(f"  Q3 Linear interpolation:   {quartile_stats['q3_linear']:10.4f}")
        print(f"  Q3 Midpoint method:        {quartile_stats['q3_midpoint']:10.4f}")
        print(f"  Q3 Manual calculation:     {quartile_stats['q3_manual']:10.4f}")
        print()
        
        # Quartile-based statistics
        print("Quartile-Based Statistics:")
        print(f"  Interquartile Range (IQR): {quartile_stats['iqr']:10.4f}")
        print(f"  Quartile Deviation (QD):   {quartile_stats['qd']:10.4f}")
        print(f"  Quartile Skewness:         {quartile_stats['quartile_skewness']:10.4f}")
        print()
        
        # Outlier analysis
        print("Outlier Analysis (IQR Method):")
        print(f"  Lower Fence (Q1 - 1.5*IQR): {quartile_stats['lower_fence']:10.4f}")
        print(f"  Upper Fence (Q3 + 1.5*IQR): {quartile_stats['upper_fence']:10.4f}")
        print(f"  Number of Outliers:         {quartile_stats['outlier_count']:10d}")
        
        if quartile_stats['outlier_count'] > 0:
            print(f"  Outlier Values:")
            for outlier in quartile_stats['outlier_values'][:10]:  # Show first 10 outliers
                print(f"    {outlier:12.4f}")
            if len(quartile_stats['outlier_values']) > 10:
                print(f"    ... and {len(quartile_stats['outlier_values']) - 10} more")
        print()
        
        # Data distribution in quartiles
        q1, q2, q3 = quartile_stats['q1_standard'], quartile_stats['q2_median'], quartile_stats['q3_standard']
        q1_range = len(algorithm_data[algorithm_data <= q1])
        q2_range = len(algorithm_data[(algorithm_data > q1) & (algorithm_data <= q2)])
        q3_range = len(algorithm_data[(algorithm_data > q2) & (algorithm_data <= q3)])
        q4_range = len(algorithm_data[algorithm_data > q3])
        
        print("Data Distribution by Quartiles:")
        print(f"  Q1 range (≤ {q1:.4f}):        {q1_range:6d} values ({q1_range/len(algorithm_data)*100:.1f}%)")
        print(f"  Q2 range ({q1:.4f} < x ≤ {q2:.4f}): {q2_range:6d} values ({q2_range/len(algorithm_data)*100:.1f}%)")
        print(f"  Q3 range ({q2:.4f} < x ≤ {q3:.4f}): {q3_range:6d} values ({q3_range/len(algorithm_data)*100:.1f}%)")
        print(f"  Q4 range (> {q3:.4f}):         {q4_range:6d} values ({q4_range/len(algorithm_data)*100:.1f}%)")
        print()
        
        print("=" * 80)
        print()

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python quartile_analysis.py <csv_file_path>")
        print("Example: python quartile_analysis.py experiment_results/execution/all_policies_execution_results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    
    try:
        generate_quartile_report(csv_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
