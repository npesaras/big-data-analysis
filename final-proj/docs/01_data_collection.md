# Data Collection and Cleaning

This document explains the first critical step in the ML pipeline: loading, inspecting, and understanding the diabetes dataset.

## Overview

The `src.data_cleaning` module provides tools for:
- Loading the PIMA Indians Diabetes dataset
- Inspecting data quality and structure
- Identifying biological impossibilities (zero values)
- Generating comprehensive data quality reports
- Detecting outliers using statistical methods

## Dataset: PIMA Indians Diabetes

### Background

The **PIMA Indians Diabetes Database** is a classic dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. It was collected to predict diabetes onset in female patients of Pima Indian heritage.

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Samples** | 768 patients |
| **Features** | 8 clinical measurements |
| **Target** | Binary (0 = No diabetes, 1 = Diabetes) |
| **Class Balance** | 500 negative (65.1%) vs 268 positive (34.9%) |
| **Missing Data** | Encoded as zeros (biologically impossible) |
| **Source** | UCI Machine Learning Repository |

### Features Description

```python
FEATURE_COLUMNS = [
    'Pregnancies',        # Number of times pregnant
    'Glucose',            # Plasma glucose concentration (mg/dL)
    'BloodPressure',      # Diastolic blood pressure (mm Hg)
    'SkinThickness',      # Triceps skin fold thickness (mm)
    'Insulin',            # 2-Hour serum insulin (mu U/ml)
    'BMI',                # Body mass index (weight in kg/(height in m)^2)
    'DiabetesPedigreeFunction',  # Genetic predisposition score
    'Age'                 # Age in years
]
```

### Feature Ranges (Observed in Dataset)

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| Pregnancies | 0 | 17 | 3.8 | 3.4 |
| Glucose | 44* | 199 | 121.7 | 30.4 |
| BloodPressure | 24* | 122 | 72.4 | 12.4 |
| SkinThickness | 7* | 99 | 29.2 | 10.5 |
| Insulin | 14* | 846 | 155.5 | 118.8 |
| BMI | 18.2* | 67.1 | 32.5 | 6.9 |
| DPF | 0.078 | 2.420 | 0.472 | 0.331 |
| Age | 21 | 81 | 33.2 | 11.8 |

*Excluding zeros

## Loading Data

### Basic Usage

```python
from src.data_cleaning import load_diabetes_data
import pandas as pd

# Load the dataset
df = load_diabetes_data()

print(f"Dataset shape: {df.shape}")
# Output: Dataset shape: (768, 9)

print(f"Features: {df.columns.tolist()}")
# Output: Features: ['Pregnancies', 'Glucose', 'BloodPressure', ...]
```

### Function Signature

```python
def load_diabetes_data(
    file_path: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load the diabetes dataset from CSV.

    Parameters
    ----------
    file_path : str, optional
        Custom path to CSV file. If None, uses config.DIABETES_DATA
    validate : bool, default=True
        Whether to validate required columns exist

    Returns
    -------
    pd.DataFrame
        Loaded dataset with all features and target

    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist at specified path
    ValueError
        If required columns are missing
    """
```

### Loading from Custom Path

```python
# Load from different location
df = load_diabetes_data(file_path="/path/to/custom_data.csv")

# Skip validation for experimental datasets
df = load_diabetes_data(validate=False)
```

## Data Inspection

### Comprehensive Inspection

```python
from src.data_cleaning import inspect_data

# Get full data report
report = inspect_data(df)

# Report includes:
# - Shape (rows, columns)
# - Memory usage
# - Column data types
# - Missing value counts
# - Basic statistics (mean, std, min, max)
# - Duplicate row detection
```

### Example Output

```
Dataset Shape: (768, 9)
Memory Usage: 55.2 KB

Column Data Types:
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64

Missing Values:
Glucose           0
BloodPressure     0
SkinThickness     0
Insulin           0
BMI               0
(All columns: 0 missing)

Duplicate Rows: 0

Basic Statistics:
              count        mean        std   min   25%   50%   75%    max
Pregnancies   768.0        3.845      3.370   0.0   1.0   3.0   6.0   17.0
Glucose       768.0      120.895     31.973   0.0  99.0 117.0 140.25 199.0
Age           768.0       33.241     11.760  21.0  24.0  29.0  41.0   81.0
...
```

## Critical Issue: Zero Values

### The Problem

The dataset encodes **missing values as zeros**, which are biologically impossible:

- **Glucose = 0**: No human can have 0 mg/dL blood glucose and be alive
- **BloodPressure = 0**: Impossible to survive without blood pressure
- **BMI = 0**: Cannot exist without body mass

### Identifying Zero Issues

```python
from src.data_cleaning import identify_zero_issues

# Get columns with impossible zeros
zero_issues = identify_zero_issues(df)

print(zero_issues)
```

**Output:**
```python
{
    'Glucose': {
        'count': 5,
        'percentage': 0.65,
        'affected_samples': [75, 182, 342, 349, 502]
    },
    'BloodPressure': {
        'count': 35,
        'percentage': 4.56,
        'affected_samples': [14, 39, 49, ...]
    },
    'SkinThickness': {
        'count': 227,
        'percentage': 29.56,
        'affected_samples': [0, 1, 2, 3, ...]
    },
    'Insulin': {
        'count': 374,
        'percentage': 48.70,
        'affected_samples': [0, 1, 3, 5, ...]
    },
    'BMI': {
        'count': 11,
        'percentage': 1.43,
        'affected_samples': [28, 105, 243, ...]
    }
}
```

### Key Insights

1. **Insulin**: 48.7% missing - most problematic feature
2. **SkinThickness**: 29.6% missing - significant data loss
3. **BloodPressure**: 4.6% missing - manageable
4. **Glucose**: 0.65% missing - critical feature, few issues
5. **BMI**: 1.4% missing - few issues

### Configuration

Columns with zero issues are defined in `src/config.py`:

```python
COLS_WITH_ZERO_ISSUES = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]
```

## Data Quality Report

### Generating Comprehensive Reports

```python
from src.data_cleaning import get_data_quality_report

# Full quality analysis
quality_report = get_data_quality_report(df)

print(quality_report['summary'])
print(quality_report['recommendations'])
```

### Report Structure

```python
{
    'summary': {
        'total_samples': 768,
        'total_features': 8,
        'total_cells': 6144,
        'missing_cells': 652,
        'missing_percentage': 10.61,
        'complete_rows': 392,
        'complete_row_percentage': 51.04
    },
    'column_quality': {
        'Glucose': {
            'missing': 5,
            'missing_pct': 0.65,
            'unique_values': 136,
            'quality_score': 'Excellent'
        },
        'Insulin': {
            'missing': 374,
            'missing_pct': 48.70,
            'unique_values': 186,
            'quality_score': 'Poor'
        },
        ...
    },
    'recommendations': [
        'Insulin: Consider dropping (48.7% missing) or advanced imputation',
        'SkinThickness: Use median/mean imputation (29.6% missing)',
        'Glucose: Critical feature - use careful imputation (0.65% missing)',
        ...
    ]
}
```

## Outlier Detection

### IQR Method

```python
from src.data_cleaning import detect_outliers_iqr

# Detect outliers using Interquartile Range
outliers = detect_outliers_iqr(df, multiplier=1.5)

print(f"Outliers detected in each column:")
for col, indices in outliers.items():
    print(f"{col}: {len(indices)} outliers")
```

### Example Results

```
Outliers detected:
Pregnancies: 159 outliers (20.7%)
Insulin: 49 outliers (6.4%)
Age: 93 outliers (12.1%)
DiabetesPedigreeFunction: 29 outliers (3.8%)
```

### Outlier Handling Strategy

In this project, we **do not remove outliers** because:

1. **Medical context**: Extreme values may indicate high-risk patients
2. **Small dataset**: 768 samples - cannot afford to lose data
3. **Real-world variability**: Diabetes affects diverse populations
4. **Model robustness**: Tree-based models handle outliers well

Instead, we use **robust preprocessing**:
- StandardScaler (less sensitive to outliers than MinMaxScaler)
- Median imputation (robust to outliers)
- Feature engineering based on clinical thresholds

## Column Statistics

### Detailed Statistics

```python
from src.data_cleaning import get_column_statistics

# Get statistics for specific columns
stats = get_column_statistics(df, columns=['Glucose', 'BMI', 'Age'])

for col, col_stats in stats.items():
    print(f"\n{col}:")
    print(f"  Mean: {col_stats['mean']:.2f}")
    print(f"  Median: {col_stats['median']:.2f}")
    print(f"  Std: {col_stats['std']:.2f}")
    print(f"  Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
    print(f"  IQR: {col_stats['iqr']:.2f}")
    print(f"  Skewness: {col_stats['skewness']:.2f}")
```

**Output:**
```
Glucose:
  Mean: 120.89
  Median: 117.00
  Std: 31.97
  Range: [0.00, 199.00]
  IQR: 41.25
  Skewness: 0.17

BMI:
  Mean: 31.99
  Median: 32.00
  Std: 7.88
  Range: [0.00, 67.10]
  IQR: 9.10
  Skewness: 0.59

Age:
  Mean: 33.24
  Median: 29.00
  Std: 11.76
  Range: [21.00, 81.00]
  IQR: 17.00
  Skewness: 1.13 (right-skewed)
```

## Clinical Validation

### Biologically Valid Ranges

Based on medical literature and dataset analysis:

```python
BIOLOGICAL_RANGES = {
    'Glucose': (44, 199),          # 44 is minimum observed non-zero
    'BloodPressure': (40, 122),    # Diastolic pressure
    'SkinThickness': (7, 99),      # Triceps fold
    'Insulin': (14, 846),          # Highly variable
    'BMI': (15.0, 67.1),          # Underweight to severe obesity
    'Age': (21, 81),               # Study population range
    'Pregnancies': (0, 17),        # Maximum observed
    'DiabetesPedigreeFunction': (0.078, 2.420)
}
```

### Validation Function

```python
def validate_input(values: dict) -> dict:
    """
    Validate patient input against biological ranges.

    Returns dict with 'valid' bool and 'warnings' list.
    """
    warnings = []

    if values['Glucose'] < 70:
        warnings.append('⚠️ Hypoglycemia risk')
    elif values['Glucose'] > 140:
        warnings.append('⚠️ Hyperglycemia detected')

    if values['BMI'] < 18.5:
        warnings.append('⚠️ Underweight')
    elif values['BMI'] >= 30:
        warnings.append('⚠️ Obesity detected')

    if values['BloodPressure'] < 60:
        warnings.append('⚠️ Low blood pressure')
    elif values['BloodPressure'] > 90:
        warnings.append('⚠️ Hypertension risk')

    return {
        'valid': len(warnings) == 0,
        'warnings': warnings
    }
```

## Best Practices

### 1. Always Inspect First

```python
# ✅ Good: Understand your data before processing
df = load_diabetes_data()
inspect_data(df)
identify_zero_issues(df)
quality_report = get_data_quality_report(df)
```

### 2. Handle Zeros Before Analysis

```python
# ❌ Bad: Analyze with zeros included
df.describe()  # Includes impossible zeros

# ✅ Good: Replace zeros with NaN first
from src.pre_processing import handle_zero_values
df_cleaned = handle_zero_values(df)
df_cleaned.describe()  # Excludes zeros
```

### 3. Document Data Decisions

```python
# Keep record of data quality issues
data_issues = {
    'insulin_missing': 374,
    'skinthickness_missing': 227,
    'strategy': 'median_imputation',
    'outliers_kept': True,
    'reason': 'Medical context - extreme values clinically relevant'
}
```

### 4. Validate Input Data

```python
# Always validate new data against training distribution
def is_within_training_range(value, column):
    min_val, max_val = BIOLOGICAL_RANGES[column]
    return min_val <= value <= max_val
```

## Common Issues and Solutions

### Issue: FileNotFoundError

```python
# ❌ Problem
df = load_diabetes_data(file_path="wrong_path.csv")
# FileNotFoundError: No such file

# ✅ Solution: Use config path
from src import config
df = load_diabetes_data(file_path=config.DIABETES_DATA)
```

### Issue: Treating Zeros as Valid

```python
# ❌ Problem: Zeros included in statistics
mean_glucose = df['Glucose'].mean()  # Includes 5 zeros

# ✅ Solution: Identify and handle zeros
zero_issues = identify_zero_issues(df)
df_cleaned = handle_zero_values(df)  # See preprocessing.md
mean_glucose = df_cleaned['Glucose'].mean()
```

### Issue: Ignoring Class Imbalance

```python
# ✅ Solution: Check class distribution
from collections import Counter
target_dist = Counter(df['Outcome'])
print(f"Class 0: {target_dist[0]} (No diabetes)")
print(f"Class 1: {target_dist[1]} (Diabetes)")
print(f"Imbalance ratio: {target_dist[0] / target_dist[1]:.2f}:1")

# Output:
# Class 0: 500 (No diabetes)
# Class 1: 268 (Diabetes)
# Imbalance ratio: 1.87:1
```

## Next Steps

After loading and understanding the data:

1. **[Exploratory Analysis](02_exploratory_analysis.md)** - Visualize distributions and correlations
2. **[Preprocessing](03_preprocessing.md)** - Handle zeros, impute, and scale features
3. **[Data Splitting](04_data_splitting.md)** - Create train/test sets with stratification

## Code Reference

Full implementation: `src/data_cleaning.py`

Key functions:
- `load_diabetes_data()` - Load CSV dataset
- `inspect_data()` - Comprehensive inspection
- `identify_zero_issues()` - Find impossible zeros
- `get_data_quality_report()` - Quality analysis
- `detect_outliers_iqr()` - Outlier detection
- `get_column_statistics()` - Detailed stats

See [API Reference](08_api_reference.md) for complete documentation.
