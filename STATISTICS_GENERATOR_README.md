# Execution Statistics Generator

This directory contains Python scripts to generate comprehensive statistical analysis from CSV data containing execution results.

## Files

1. **`generate_execution_statistics.py`** - Complete script that generates both detailed statistics and summary table files
2. **`simple_statistics_generator.py`** - Command-line script for interactive use with any CSV file
3. **`STATISTICS_GENERATOR_README.md`** - This documentation file

## Usage

### Method 1: Complete Script (Recommended for batch processing)

```bash
python generate_execution_statistics.py
```

This will:
- Read from: `/home/dongmingwang/project/Expert_RL/experiment_results/execution/all_policies_execution_results.csv`
- Generate: `execution_statistics_generated.txt` and `execution_summary_table_generated.txt`

### Method 2: Simple Command-line Script (Flexible)

```bash
python simple_statistics_generator.py <path_to_csv_file>
```

Example:
```bash
python simple_statistics_generator.py experiment_results/execution/all_policies_execution_results.csv
```

## Expected CSV Format

The scripts expect a CSV file with the following columns:
- `algorithm` - Algorithm identifier (e.g., 'llm_policy', 'mapg_ep250', etc.)
- `episode_return` - The metric to analyze (episode returns in this case)

Other columns are ignored for statistical analysis.

## Generated Statistics

### Basic Statistics
- **Count**: Number of data points
- **Mean**: Average value
- **Median**: Middle value (50th percentile)
- **Standard Deviation**: Measure of variability
- **Coefficient of Variation**: Relative variability (std/mean)

### Range Statistics
- **Minimum/Maximum**: Data range bounds
- **Range**: Difference between max and min
- **Q25/Q75**: 25th and 75th percentiles (quartiles)
- **IQR**: Interquartile range (Q75 - Q25)

### Distribution Shape
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness
- **Median Absolute Deviation**: Robust measure of variability

### Interpretation
- **Distribution symmetry**: Symmetric, left-skewed, or right-skewed
- **Tail behavior**: Light tails, heavy tails, or normal tails
- **Variability**: Low, moderate, or high relative to mean

## Sample Output

The scripts generate output in the same format as the original `execution_statistics.txt` and `execution_summary_table.txt` files, including:

1. **Detailed Statistics**: Comprehensive analysis for each algorithm
2. **Comparative Analysis**: Performance and consistency rankings
3. **Summary Table**: Condensed overview of all algorithms

## Dependencies

Required Python packages:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Statistical functions

Install with:
```bash
pip install pandas numpy scipy
```

## Customization

### For Different CSV Files

Modify the file paths in `generate_execution_statistics.py`:
```python
csv_file = "path/to/your/data.csv"
stats_output = "path/to/your/statistics.txt"
summary_output = "path/to/your/summary.txt"
```

### For Different Column Names

In `simple_statistics_generator.py`, you can modify the column names:
```python
generate_statistics_report(csv_file, algorithm_col='your_algorithm_column', value_col='your_value_column')
```

### For Different Algorithm Names

Add name mappings in `generate_execution_statistics.py`:
```python
name_mapping = {
    'your_algorithm': 'Your Display Name',
    # ... existing mappings
}
```

## Example Results

Based on the execution data, the scripts show:
- **Best Performer**: mapg_ep1500 (Mean: 1.6165)
- **Most Consistent**: mapg_no_expert (CV: 0.1102)
- **Worst Performer**: llm_policy (Mean: 0.3206)

The analysis reveals that expert policy methods generally outperform LLM-only approaches, with varying degrees of consistency across different training configurations.
