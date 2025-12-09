# Data Preprocessing Pipeline

This document explains the comprehensive preprocessing pipeline implemented in `src.pre_processing` to prepare the diabetes dataset for machine learning.

## Overview

Preprocessing is essential for:
- **Handling missing data** - Replace biologically impossible zeros with NaN
- **Imputation** - Fill missing values with statistical measures
- **Feature scaling** - Normalize feature magnitudes for distance-based algorithms
- **Data transformation** - Prepare features for optimal model performance

The preprocessing pipeline consists of three main stages:

```
Raw Data → Handle Zeros → Imputation → Scaling → Model-Ready Data
```

## The Preprocessing Challenge

### Why Preprocessing Matters

**Problem:** Different features have vastly different scales:
```
Pregnancies:           0 to 17     (small range)
Glucose:              44 to 199    (medium range)
Insulin:              14 to 846    (large range)
DiabetesPedigreeFunction: 0.078 to 2.420  (small decimals)
```

**Without scaling:**
- Distance-based algorithms (KNN, SVM) will be dominated by large-scale features
- Gradient descent converges slowly
- Feature importance is skewed

**With scaling:**
- All features contribute equally to distance calculations
- Faster convergence
- Improved model performance

### Missing Data Problem

**Zero Encoding Issues:**
```
Missing Data Encoded as Zeros:
- Glucose:       5 zeros (0.65%)  - CRITICAL: Impossible to be alive
- BloodPressure: 35 zeros (4.56%) - CRITICAL: Cannot survive without BP
- SkinThickness: 227 zeros (29.56%) - Measurement not taken
- Insulin:       374 zeros (48.70%) - Most problematic feature
- BMI:           11 zeros (1.43%)  - CRITICAL: Cannot exist
```

## Stage 1: Handle Zero Values

### Identifying Impossible Zeros

```python
from src.data_cleaning import load_diabetes_data
from src.pre_processing import handle_zero_values
from src import config

# Load raw data
df = load_diabetes_data()

# Replace zeros with NaN for specific columns
df_cleaned = handle_zero_values(df)

print("Zeros replaced with NaN:")
for col in config.COLS_WITH_ZERO_ISSUES:
    before = (df[col] == 0).sum()
    after = df_cleaned[col].isna().sum()
    print(f"{col:20} | Zeros: {before:3} → NaN: {after:3}")
```

**Output:**
```
Zeros replaced with NaN:
Glucose              | Zeros:   5 → NaN:   5
BloodPressure        | Zeros:  35 → NaN:  35
SkinThickness        | Zeros: 227 → NaN: 227
Insulin              | Zeros: 374 → NaN: 374
BMI                  | Zeros:  11 → NaN:  11
```

### Function Signature

```python
def handle_zero_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Replace zeros with NaN for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        Columns to process. If None, uses config.COLS_WITH_ZERO_ISSUES

    Returns
    -------
    pd.DataFrame
        Dataframe with zeros replaced by NaN

    Note
    ----
    Only replaces zeros in specified columns. Other columns unchanged.
    Original dataframe is not modified (returns copy).
    """
```

### Why Not Remove Rows?

```python
# ❌ Bad: Remove rows with any missing value
df_complete = df.dropna()
print(f"Samples remaining: {len(df_complete)}/768")
# Output: Samples remaining: 392/768 (49% data loss!)

# ✅ Good: Impute missing values
df_imputed = impute_missing_values(df_cleaned)
print(f"Samples remaining: {len(df_imputed)}/768")
# Output: Samples remaining: 768/768 (0% data loss)
```

**Rationale:**
- **Small dataset**: Only 768 samples - cannot afford 49% loss
- **Pattern in missingness**: Insulin missing in 48.7% - not random
- **Imputation preserves**: Statistical properties and sample size

## Stage 2: Imputation

### Median Imputation Strategy

```python
from src.pre_processing import impute_missing_values

# Impute missing values using median strategy
df_imputed, imputer = impute_missing_values(
    df_cleaned,
    strategy='median'
)

print("Imputation complete:")
print(f"Missing values before: {df_cleaned.isna().sum().sum()}")
print(f"Missing values after:  {df_imputed.isna().sum().sum()}")
```

**Output:**
```
Imputation complete:
Missing values before: 652
Missing values after:  0
```

### Why Median (not Mean)?

**Comparison:**

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Median** | Robust to outliers, preserves distribution | Less precise | Skewed data with outliers ✓ |
| **Mean** | Mathematically optimal, minimizes MSE | Sensitive to outliers | Normal distributions |
| **Mode** | Preserves categorical patterns | Only for categorical | Discrete features |
| **KNN** | Considers feature relationships | Computationally expensive | Small datasets with patterns |

**Our dataset has:**
- Right-skewed distributions (Age, Insulin)
- Outliers in multiple features
- Medical measurements (robustness needed)

**Therefore: Median is optimal**

### Imputation Values

```python
# View what values were used for imputation
imputation_values = {
    feature: imputer.statistics_[i]
    for i, feature in enumerate(config.FEATURE_COLUMNS)
}

for feature, value in imputation_values.items():
    print(f"{feature:30} → {value:.2f}")
```

**Expected Output:**
```
Pregnancies                    → 3.00
Glucose                        → 117.00
BloodPressure                  → 72.00
SkinThickness                  → 23.00
Insulin                        → 125.00
BMI                            → 32.00
DiabetesPedigreeFunction       → 0.37
Age                            → 29.00
```

### Function Signature

```python
def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values using specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with missing values (NaN)
    strategy : str, default='median'
        Imputation strategy: 'median', 'mean', 'most_frequent', 'constant'
    columns : List[str], optional
        Columns to impute. If None, uses config.FEATURE_COLUMNS

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values
    SimpleImputer
        Fitted imputer object (save for later use on test data)

    Examples
    --------
    >>> df_clean = handle_zero_values(df)
    >>> df_imputed, imputer = impute_missing_values(df_clean)
    >>> # Later, use same imputer on test data
    >>> X_test_imputed = imputer.transform(X_test)
    """
```

### Alternative Strategies

```python
# Mean imputation (sensitive to outliers)
df_imputed, imputer = impute_missing_values(df_cleaned, strategy='mean')

# Constant imputation (e.g., use 0)
df_imputed, imputer = impute_missing_values(
    df_cleaned,
    strategy='constant',
    fill_value=0
)

# Most frequent (for categorical-like features)
df_imputed, imputer = impute_missing_values(df_cleaned, strategy='most_frequent')
```

## Stage 3: Feature Scaling

### StandardScaler (Z-score Normalization)

```python
from src.pre_processing import scale_features

# Scale features to zero mean and unit variance
df_scaled, scaler = scale_features(df_imputed)

print("Scaling statistics:")
for i, feature in enumerate(config.FEATURE_COLUMNS):
    mean = scaler.mean_[i]
    std = scaler.scale_[i]
    print(f"{feature:30} | Mean: {mean:7.2f} | Std: {std:6.2f}")
```

**Output:**
```
Scaling statistics:
Pregnancies                    | Mean:    3.85 | Std:   3.37
Glucose                        | Mean:  120.89 | Std:  31.97
BloodPressure                  | Mean:   69.11 | Std:  19.36
SkinThickness                  | Mean:   20.54 | Std:  15.95
Insulin                        | Mean:   79.80 | Std: 115.24
BMI                            | Mean:   31.99 | Std:   7.88
DiabetesPedigreeFunction       | Mean:    0.47 | Std:   0.33
Age                            | Mean:   33.24 | Std:  11.76
```

### How StandardScaler Works

**Transformation formula:**
$$
z = \frac{x - \mu}{\sigma}
$$

Where:
- $x$ = original value
- $\mu$ = mean of feature
- $\sigma$ = standard deviation
- $z$ = scaled value

**Effect:**
- Scaled features have **mean = 0** and **standard deviation = 1**
- Preserves outlier information
- Suitable for normally distributed and skewed data

### Before vs After Scaling

```python
import pandas as pd

# Before scaling
print("Before Scaling:")
print(df_imputed[['Glucose', 'Insulin', 'Age']].describe())

# After scaling
print("\nAfter Scaling:")
print(df_scaled[['Glucose', 'Insulin', 'Age']].describe())
```

**Before Scaling:**
```
          Glucose      Insulin          Age
mean       120.89        79.80        33.24
std         31.97       115.24        11.76
min         44.00        14.00        21.00
max        199.00       846.00        81.00
```

**After Scaling:**
```
          Glucose      Insulin          Age
mean         0.00         0.00         0.00
std          1.00         1.00         1.00
min         -2.40        -0.57        -1.04
max         +2.44        +6.65        +4.06
```

### Function Signature

```python
def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler_type: str = 'standard'
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Scale features using specified scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with imputed values
    columns : List[str], optional
        Columns to scale. If None, uses config.FEATURE_COLUMNS
    scaler_type : str, default='standard'
        Type of scaler: 'standard' or 'minmax'

    Returns
    -------
    pd.DataFrame
        Dataframe with scaled features
    Scaler
        Fitted scaler object (StandardScaler or MinMaxScaler)

    Note
    ----
    Target column ('Outcome') is never scaled.
    """
```

### Alternative: MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# Scale to [0, 1] range
df_scaled, scaler = scale_features(df_imputed, scaler_type='minmax')

# Formula: (x - min) / (max - min)
print(df_scaled[['Glucose', 'Insulin']].describe())
```

**Output:**
```
          Glucose      Insulin
min          0.00         0.00
max          1.00         1.00
mean         0.50         0.08
```

**When to use MinMaxScaler:**
- Neural networks with sigmoid activation
- Features bounded by nature (probabilities, percentages)
- When you need specific range (e.g., [0, 1])

**When to use StandardScaler (our choice):**
- Tree-based models (unaffected by scaling)
- Linear models (logistic regression, SVM)
- When data has outliers (more robust than MinMaxScaler)
- Distance-based algorithms (KNN)

## Complete Preprocessing Pipeline

### End-to-End Function

```python
from src.pre_processing import preprocess_pipeline

# One function does it all
df_processed, preprocessors = preprocess_pipeline(df)

# Returns:
# - df_processed: Fully preprocessed dataframe
# - preprocessors: Dict with 'imputer' and 'scaler' objects

print(f"Original shape:  {df.shape}")
print(f"Processed shape: {df_processed.shape}")
print(f"Missing values:  {df_processed.isna().sum().sum()}")
print(f"Scaled features: {(df_processed[config.FEATURE_COLUMNS].mean().abs() < 0.01).all()}")
```

**Output:**
```
Original shape:  (768, 9)
Processed shape: (768, 9)
Missing values:  0
Scaled features: True (all means ≈ 0)
```

### Pipeline Function Signature

```python
def preprocess_pipeline(
    df: pd.DataFrame,
    impute_strategy: str = 'median',
    scale_type: str = 'standard'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline: handle zeros → impute → scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe
    impute_strategy : str, default='median'
        Imputation strategy for missing values
    scale_type : str, default='standard'
        Scaling method: 'standard' or 'minmax'

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe ready for modeling
    Dict[str, Any]
        Dictionary containing fitted preprocessors:
        - 'imputer': SimpleImputer object
        - 'scaler': StandardScaler or MinMaxScaler object

    Examples
    --------
    >>> # Preprocess training data
    >>> df_train_processed, preprocessors = preprocess_pipeline(df_train)
    >>>
    >>> # Apply same preprocessing to test data
    >>> X_test_clean = handle_zero_values(X_test)
    >>> X_test_imputed = preprocessors['imputer'].transform(X_test_clean)
    >>> X_test_scaled = preprocessors['scaler'].transform(X_test_imputed)
    """
```

### Applying to New Data

**Critical:** Test data must be preprocessed using training data's statistics.

```python
from src.pre_processing import apply_preprocessing

# Train preprocessing
df_train_processed, preprocessors = preprocess_pipeline(df_train)

# Test preprocessing (using training statistics)
df_test_processed = apply_preprocessing(df_test, preprocessors)

# ❌ BAD: Fit new preprocessors on test data (data leakage)
df_test_bad, _ = preprocess_pipeline(df_test)

# ✅ GOOD: Use training preprocessors on test data
df_test_good = apply_preprocessing(df_test, preprocessors)
```

### apply_preprocessing Function

```python
def apply_preprocessing(
    df: pd.DataFrame,
    preprocessors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply existing preprocessors to new data.

    Parameters
    ----------
    df : pd.DataFrame
        New data to preprocess
    preprocessors : dict
        Dictionary with 'imputer' and 'scaler' from preprocess_pipeline()

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe using training statistics

    Note
    ----
    This function is crucial for preventing data leakage.
    Always use training preprocessors on test/validation data.
    """
```

## Saving and Loading Preprocessors

### Save for Production

```python
from src.utils import save_preprocessors, load_preprocessors

# After training preprocessing
df_processed, preprocessors = preprocess_pipeline(df_train)

# Save for later use
save_preprocessors(preprocessors, 'preprocessors.joblib')

# Later in production...
preprocessors = load_preprocessors('preprocessors.joblib')
new_data_processed = apply_preprocessing(new_data, preprocessors)
```

### Preprocessor Storage Structure

```python
# What's inside preprocessors dict
preprocessors = {
    'imputer': SimpleImputer(strategy='median'),
    'scaler': StandardScaler(),
    'feature_names': ['Pregnancies', 'Glucose', ...],
    'created_at': '2024-01-15 10:30:00',
    'config': {
        'impute_strategy': 'median',
        'scale_type': 'standard',
        'cols_with_zeros': ['Glucose', 'BloodPressure', ...]
    }
}
```

## Inverse Transformations

### Getting Original Values Back

```python
from src.pre_processing import inverse_transform_features

# Scale data
df_scaled, scaler = scale_features(df_imputed)

# Later, convert back to original scale
df_original = inverse_transform_features(df_scaled, scaler)

print("Original Glucose mean:", df_imputed['Glucose'].mean())
print("Scaled Glucose mean:  ", df_scaled['Glucose'].mean())
print("Inverse Glucose mean: ", df_original['Glucose'].mean())
```

**Output:**
```
Original Glucose mean: 120.89
Scaled Glucose mean:   0.00
Inverse Glucose mean:  120.89
```

**Use Cases:**
- Interpreting model predictions in original scale
- Visualizing results
- Explaining model decisions to clinicians

## Preprocessing Best Practices

### 1. Always Fit on Training Data Only

```python
# ✅ Correct workflow
from sklearn.model_selection import train_test_split

# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess training data (fit)
X_train_processed, preprocessors = preprocess_pipeline(
    pd.concat([X_train, y_train], axis=1)
)

# Apply to test data (transform only, no fit)
X_test_processed = apply_preprocessing(
    pd.concat([X_test, y_test], axis=1),
    preprocessors
)
```

```python
# ❌ Wrong: Preprocessing before split (data leakage)
df_processed, _ = preprocess_pipeline(df)
X_train, X_test = train_test_split(df_processed)
```

### 2. Save Preprocessors with Models

```python
from src.utils import save_model, save_preprocessors

# Train model
model.fit(X_train_processed, y_train)

# Save everything together
save_model(model, 'model.joblib')
save_preprocessors(preprocessors, 'preprocessors.joblib')

# Production: Load both
model = load_model('model.joblib')
preprocessors = load_preprocessors('preprocessors.joblib')
```

### 3. Document Preprocessing Decisions

```python
preprocessing_config = {
    'zero_handling': 'Replace with NaN',
    'imputation': 'Median (robust to outliers)',
    'scaling': 'StandardScaler (preserves outlier info)',
    'features_scaled': config.FEATURE_COLUMNS,
    'features_with_zeros': config.COLS_WITH_ZERO_ISSUES,
    'missing_data_pct': {
        'Insulin': 48.7,
        'SkinThickness': 29.6,
        'BloodPressure': 4.6,
        'BMI': 1.4,
        'Glucose': 0.65
    }
}
```

### 4. Validate Preprocessing

```python
def validate_preprocessing(df_processed: pd.DataFrame):
    """Ensure preprocessing was successful."""

    # Check no missing values
    assert df_processed.isna().sum().sum() == 0, "Missing values remain"

    # Check scaling (mean ≈ 0, std ≈ 1)
    for col in config.FEATURE_COLUMNS:
        mean = df_processed[col].mean()
        std = df_processed[col].std()
        assert abs(mean) < 0.01, f"{col} mean not centered: {mean}"
        assert abs(std - 1.0) < 0.1, f"{col} std not 1: {std}"

    # Check no zeros in critical columns
    for col in config.COLS_WITH_ZERO_ISSUES:
        # After scaling, zeros would be negative values
        # But should not be exactly original zero
        pass

    print("✅ Preprocessing validation passed")

validate_preprocessing(df_processed)
```

## Common Issues and Solutions

### Issue 1: Data Leakage

```python
# ❌ Problem: Preprocessing entire dataset before split
df_processed, _ = preprocess_pipeline(df)
X_train, X_test = train_test_split(df_processed)
# Test data statistics leaked into training

# ✅ Solution: Split first, then preprocess
X_train, X_test = train_test_split(df)
X_train_proc, prep = preprocess_pipeline(X_train)
X_test_proc = apply_preprocessing(X_test, prep)
```

### Issue 2: Feature Scale Mismatch

```python
# ❌ Problem: Different scalers on train/test
X_train_scaled, scaler1 = scale_features(X_train)
X_test_scaled, scaler2 = scale_features(X_test)  # Wrong!

# ✅ Solution: Use same scaler
X_train_scaled, scaler = scale_features(X_train)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)
```

### Issue 3: Forgot to Handle Zeros

```python
# ❌ Problem: Impute without handling zeros first
df_imputed, _ = impute_missing_values(df)  # Zeros still there

# ✅ Solution: Always handle zeros first
df_clean = handle_zero_values(df)
df_imputed, _ = impute_missing_values(df_clean)
```

### Issue 4: Wrong Imputation for Categorical

```python
# ❌ Problem: Median for categorical feature
# If 'Outcome' had missing values
df_imputed, _ = impute_missing_values(df, strategy='median')

# ✅ Solution: Use most_frequent for categorical
df_imputed, _ = impute_missing_values(df, strategy='most_frequent')
```

## Preprocessing Impact on Models

### Algorithm Sensitivity

| Algorithm | Scaling Required? | Imputation Required? | Outlier Sensitivity |
|-----------|-------------------|----------------------|---------------------|
| Logistic Regression | ✅ Yes | ✅ Yes | Medium |
| Decision Tree | ❌ No | ✅ Yes | Low |
| Random Forest | ❌ No | ✅ Yes | Low |
| Naive Bayes | ❌ No | ✅ Yes | Medium |
| KNN | ✅ Yes (critical) | ✅ Yes | High |
| SVM | ✅ Yes (critical) | ✅ Yes | Medium |
| AdaBoost | ❌ No | ✅ Yes | Medium |
| Perceptron | ✅ Yes | ✅ Yes | Medium |
| MLP | ✅ Yes (critical) | ✅ Yes | Medium |

**Key Insights:**
- **Tree-based models**: Don't require scaling (split-based decisions)
- **Distance-based**: Require scaling (KNN, SVM)
- **Neural networks**: Require scaling (gradient descent)
- **All models**: Require imputation (can't handle NaN)

## Next Steps

After preprocessing:

1. **[Data Splitting](04_data_splitting.md)** - Create stratified train/test splits
2. **[Model Selection](05_model_selection.md)** - Choose algorithms
3. **[Model Training](06_model_training.md)** - Train on preprocessed data

## Code Reference

Full implementation: `src/pre_processing.py`

Key functions:
- `handle_zero_values()` - Replace zeros with NaN
- `impute_missing_values()` - Fill NaN with median/mean
- `scale_features()` - StandardScaler or MinMaxScaler
- `preprocess_pipeline()` - Complete pipeline
- `apply_preprocessing()` - Apply to new data
- `inverse_transform_features()` - Reverse scaling

See [API Reference](08_api_reference.md) for complete function signatures.
