# Data Splitting Strategy

This document explains the data splitting methodology in `src.data_splitting` for creating robust train/test sets and implementing cross-validation.

## Overview

Proper data splitting is critical for:
- **Unbiased evaluation** - Test set never seen during training
- **Generalization assessment** - How well model performs on new data
- **Preventing data leakage** - No information flows from test to train
- **Stratification** - Maintaining class balance across splits
- **Cross-validation** - Robust performance estimation

## Why Splitting Matters

### The Problem: Overfitting

**Without proper splitting:**
```python
# ❌ Training and testing on same data
model.fit(X, y)
score = model.score(X, y)  # Overly optimistic!
print(f"Accuracy: {score:.1%}")  # 100% - but meaningless
```

**With proper splitting:**
```python
# ✅ Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.1%}")  # Realistic performance
```

### Medical Context Importance

In diabetes prediction:
- **False Negative**: Patient has diabetes but predicted negative → dangerous
- **False Positive**: Patient doesn't have diabetes but predicted positive → unnecessary tests
- **Evaluation must be realistic** to make clinical decisions

## Train-Test Split

### Basic Split Function

```python
from src.data_splitting import split_data
from src.data_cleaning import load_diabetes_data
from src.pre_processing import preprocess_pipeline

# Load and preprocess
df = load_diabetes_data()
df_processed, preprocessors = preprocess_pipeline(df)

# Split into train and test
X_train, X_test, y_train, y_test = split_data(
    df_processed,
    test_size=0.2,
    random_state=42
)

print(f"Training samples:   {len(X_train)}")
print(f"Test samples:       {len(X_test)}")
print(f"Training ratio:     {len(X_train)/len(df):.1%}")
print(f"Test ratio:         {len(X_test)/len(df):.1%}")
```

**Output:**
```
Training samples:   614
Test samples:       154
Training ratio:     80.0%
Test ratio:         20.0%
```

### Function Signature

```python
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with features and 'Outcome' column
    test_size : float, default=0.2
        Proportion of data for test set (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to maintain class balance in splits

    Returns
    -------
    X_train : pd.DataFrame
        Training features (80% of data)
    X_test : pd.DataFrame
        Test features (20% of data)
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels

    Note
    ----
    Always uses 'Outcome' as target column.
    Stratification is recommended for imbalanced datasets.
    """
```

### Choosing Split Ratio

| Split Ratio | Training Size | Test Size | Use Case |
|-------------|---------------|-----------|----------|
| 80/20 | 614 samples | 154 samples | **Standard choice** ✓ |
| 70/30 | 537 samples | 231 samples | More test data for robust evaluation |
| 90/10 | 691 samples | 77 samples | Large datasets, need more training data |
| 60/40 | 460 samples | 308 samples | Small datasets, prioritize evaluation |

**Our choice: 80/20**
- Sufficient test samples (154) for reliable evaluation
- Enough training samples (614) for learning patterns
- Industry standard for datasets ~1000 samples

## Stratified Sampling

### Why Stratification?

**Problem without stratification:**
```python
# Random split (no stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=None  # No stratification
)

print("Train class distribution:", y_train.value_counts())
print("Test class distribution: ", y_test.value_counts())
```

**Possible bad outcome:**
```
Train class distribution: 0: 420, 1: 194  (68.4% vs 31.6%)
Test class distribution:  0:  80, 1:  74  (52.0% vs 48.0%)  ← Imbalanced!
```

**With stratification:**
```python
# Stratified split (maintains class balance)
X_train, X_test, y_train, y_test = split_data(
    df_processed,
    test_size=0.2,
    stratify=True
)

print("Train class distribution:", y_train.value_counts())
print("Test class distribution: ", y_test.value_counts())
```

**Guaranteed outcome:**
```
Train class distribution: 0: 400, 1: 214  (65.1% vs 34.9%)  ✓
Test class distribution:  0: 100, 1:  54  (65.1% vs 34.9%)  ✓
```

### Stratification Benefits

1. **Consistent evaluation**: Test set mirrors real-world distribution
2. **Fairness**: Both classes equally represented
3. **Reduced variance**: More stable performance estimates
4. **Better comparison**: Fair comparison across models

### Implementation Details

```python
from sklearn.model_selection import train_test_split

# How split_data() implements stratification
X = df[config.FEATURE_COLUMNS]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Use target for stratification
)
```

## Cross-Validation

### Why Cross-Validation?

**Single split limitation:**
- Performance depends on which samples landed in test set
- One "unlucky" split might overestimate or underestimate performance
- No confidence interval on metrics

**Cross-validation solution:**
- Multiple train/test splits
- Average performance across splits
- More robust estimate with confidence bounds

### K-Fold Cross-Validation

```python
from src.data_splitting import create_cross_validation_folds

# Create 10-fold stratified CV
cv_folds = create_cross_validation_folds(
    df_processed,
    n_folds=10,
    random_state=42
)

print(f"Number of folds: {cv_folds.get_n_splits()}")
print(f"Fold type: Stratified K-Fold")
```

**How it works:**
```
Dataset (768 samples)
├── Fold 1: Train on 691, Test on 77
├── Fold 2: Train on 691, Test on 77
├── Fold 3: Train on 691, Test on 77
├── Fold 4: Train on 691, Test on 77
├── Fold 5: Train on 691, Test on 77
├── Fold 6: Train on 691, Test on 77
├── Fold 7: Train on 691, Test on 77
├── Fold 8: Train on 691, Test on 77
├── Fold 9: Train on 691, Test on 77
└── Fold 10: Train on 691, Test on 77

Every sample appears in test set exactly once.
Average of 10 test scores = robust performance estimate.
```

### Function Signature

```python
def create_cross_validation_folds(
    df: pd.DataFrame,
    n_folds: int = 10,
    random_state: int = 42,
    stratified: bool = True
) -> Union[StratifiedKFold, KFold]:
    """
    Create cross-validation fold splitter.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe
    n_folds : int, default=10
        Number of folds (typically 5 or 10)
    random_state : int, default=42
        Random seed for reproducibility
    stratified : bool, default=True
        Use stratified folds (maintains class balance)

    Returns
    -------
    StratifiedKFold or KFold
        Sklearn CV splitter object

    Examples
    --------
    >>> cv = create_cross_validation_folds(df, n_folds=10)
    >>> for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    >>>     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    >>>     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    >>>     model.fit(X_train, y_train)
    >>>     score = model.score(X_test, y_test)
    >>>     print(f"Fold {fold}: Accuracy = {score:.3f}")
    """
```

### Using Cross-Validation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create model
model = LogisticRegression(max_iter=1000, random_state=42)

# Create CV folds
cv = create_cross_validation_folds(df_processed, n_folds=10)

# Evaluate using CV
X = df_processed[config.FEATURE_COLUMNS]
y = df_processed['Outcome']

scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

print(f"CV Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

**Output:**
```
CV Scores: [0.78, 0.75, 0.79, 0.77, 0.76, 0.78, 0.77, 0.79, 0.76, 0.75]
Mean Accuracy: 0.770 (+/- 0.026)
```

**Interpretation:**
- **Mean**: Expected accuracy on unseen data
- **±2 std**: 95% confidence interval
- Model achieves 77.0% ± 2.6% accuracy

### Choosing Number of Folds

| K Value | Pros | Cons | Use Case |
|---------|------|------|----------|
| **k=5** | Fast, less computation | Higher variance | Quick experiments |
| **k=10** | Good bias-variance tradeoff | Moderate computation | **Standard choice** ✓ |
| **k=20** | Lower variance | More computation | Small datasets |
| **k=n (LOOCV)** | No randomness, deterministic | Very slow | Tiny datasets (<100 samples) |

**Our choice: k=10**
- Industry standard
- Good balance between bias and variance
- Reasonable computation time
- Each fold has ~77 test samples (statistically significant)

## Split Information and Validation

### Inspecting Split Quality

```python
from src.data_splitting import get_split_info

# After splitting
X_train, X_test, y_train, y_test = split_data(df_processed)

# Get detailed split information
split_info = get_split_info(X_train, X_test, y_train, y_test)

print("Split Information:")
print(f"Train samples:     {split_info['train_samples']}")
print(f"Test samples:      {split_info['test_samples']}")
print(f"Train ratio:       {split_info['train_ratio']:.1%}")
print(f"Test ratio:        {split_info['test_ratio']:.1%}")
print(f"\nTrain class dist:  {split_info['train_class_dist']}")
print(f"Test class dist:   {split_info['test_class_dist']}")
print(f"\nBalance maintained: {split_info['stratification_ok']}")
```

**Output:**
```
Split Information:
Train samples:     614
Test samples:      154
Train ratio:       80.0%
Test ratio:        20.0%

Train class dist:  {0: 400, 1: 214}
Test class dist:   {0: 100, 1: 54}

Balance maintained: True ✓
```

### Validating Splits

```python
from src.data_splitting import validate_split

# Check if split is valid
is_valid, issues = validate_split(X_train, X_test, y_train, y_test)

if is_valid:
    print("✅ Split validation passed")
else:
    print("❌ Split issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

**Validation checks:**
- No sample appears in both train and test
- Test size matches expected proportion
- Class distribution maintained (if stratified)
- No data leakage (indices don't overlap)
- Sufficient samples in both sets

### Function Signature

```python
def get_split_info(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:
    """
    Get comprehensive information about data split.

    Returns
    -------
    dict
        Dictionary containing:
        - 'train_samples': Number of training samples
        - 'test_samples': Number of test samples
        - 'total_samples': Total samples
        - 'train_ratio': Proportion in training set
        - 'test_ratio': Proportion in test set
        - 'train_class_dist': Class distribution in training
        - 'test_class_dist': Class distribution in test
        - 'train_class_pct': Class percentages in training
        - 'test_class_pct': Class percentages in test
        - 'stratification_ok': Boolean if distributions match
    """
```

## Advanced Splitting Strategies

### Time-Based Split (Sequential)

**For time-series data** (not applicable to diabetes dataset):

```python
# Train on earlier data, test on later data
split_point = int(len(df) * 0.8)
X_train = df[:split_point][features]
X_test = df[split_point:][features]
y_train = df[:split_point]['Outcome']
y_test = df[split_point:]['Outcome']
```

**When to use:**
- Data has temporal ordering
- Want to predict future based on past
- Avoid look-ahead bias

### Group-Based Split

**For grouped data** (e.g., multiple records per patient):

```python
from sklearn.model_selection import GroupShuffleSplit

# Ensure same patient doesn't appear in both train and test
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

### Nested Cross-Validation

**For hyperparameter tuning + evaluation:**

```python
from sklearn.model_selection import GridSearchCV

# Outer loop: Evaluation
outer_cv = create_cross_validation_folds(df, n_folds=5)

# Inner loop: Hyperparameter tuning
inner_cv = create_cross_validation_folds(df, n_folds=3)

# Grid search with nested CV
model = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(model, param_grid, cv=inner_cv)

# Evaluate with outer CV
scores = cross_val_score(grid_search, X, y, cv=outer_cv)
print(f"Nested CV Score: {scores.mean():.3f}")
```

## Best Practices

### 1. Always Stratify (Unless Good Reason Not To)

```python
# ✅ Default: Use stratification
X_train, X_test, y_train, y_test = split_data(df, stratify=True)

# ❌ Only disable for special cases
X_train, X_test, y_train, y_test = split_data(df, stratify=False)
```

### 2. Set Random Seed for Reproducibility

```python
# ✅ Reproducible results
X_train, X_test, y_train, y_test = split_data(df, random_state=42)

# ❌ Different results each run
X_train, X_test, y_train, y_test = split_data(df, random_state=None)
```

### 3. Split Before Preprocessing

```python
# ✅ Correct order: Split → Preprocess
df_raw = load_diabetes_data()
X_train, X_test, y_train, y_test = split_data(df_raw)

# Preprocess each split separately
X_train_proc, prep = preprocess_pipeline(X_train)
X_test_proc = apply_preprocessing(X_test, prep)

# ❌ Wrong order: Preprocess → Split (data leakage)
df_processed, _ = preprocess_pipeline(df_raw)
X_train, X_test, y_train, y_test = split_data(df_processed)
```

### 4. Use Cross-Validation for Model Selection

```python
# ✅ CV for comparing models
models = create_all_models()
cv = create_cross_validation_folds(df)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name}: {scores.mean():.3f}")

# Then train best model on full training set
best_model.fit(X_train, y_train)

# Final evaluation on held-out test set
test_score = best_model.score(X_test, y_test)
```

### 5. Reserve Test Set for Final Evaluation Only

```python
# ✅ Correct workflow
# 1. Split into train and test
X_train, X_test, y_train, y_test = split_data(df)

# 2. Use CV on training set for model selection
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

# 3. Train final model on full training set
model.fit(X_train, y_train)

# 4. Evaluate ONCE on test set
final_score = model.score(X_test, y_test)

# ❌ Wrong: Multiple evaluations on test set
for param in param_values:
    model.set_params(param=param)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Overfitting to test set!
```

## Common Issues and Solutions

### Issue 1: Imbalanced Test Set

```python
# Problem: Random split creates imbalanced test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Train: {y_train.value_counts()}")
print(f"Test:  {y_test.value_counts()}")
# Test might have 70% class 0 instead of 65%

# Solution: Use stratification
X_train, X_test, y_train, y_test = split_data(df, stratify=True)
```

### Issue 2: Data Leakage from Preprocessing

```python
# ❌ Problem: Scaler fitted on full dataset
df_scaled, scaler = scale_features(df)
X_train, X_test = split_data(df_scaled)
# Test data statistics leaked into scaler

# ✅ Solution: Preprocess after splitting
X_train, X_test = split_data(df)
X_train_scaled, scaler = scale_features(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Issue 3: Too Small Test Set

```python
# ❌ Problem: Test set too small for reliable evaluation
X_train, X_test, y_train, y_test = split_data(df, test_size=0.05)
print(f"Test samples: {len(X_test)}")  # 38 samples - unreliable

# ✅ Solution: Use standard 20% or cross-validation
X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
# Or use 10-fold CV for robust estimate
```

### Issue 4: Forgetting to Shuffle

```python
# ❌ Problem: Data is ordered (all class 0, then all class 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # Keeps original order
)
# Train on class 0, test on class 1!

# ✅ Solution: Always shuffle (default in split_data)
X_train, X_test, y_train, y_test = split_data(df)  # shuffle=True by default
```

## Evaluation Metrics by Split

### Metrics on Training Set

```python
# Training metrics (should be high)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.1%}")
```

**Purpose:**
- Check if model is learning
- Detect underfitting (low training score)
- Baseline for overfitting detection

### Metrics on Cross-Validation

```python
# CV metrics (realistic estimate)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
print(f"CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
```

**Purpose:**
- Model selection
- Hyperparameter tuning
- Robust performance estimate
- Confidence intervals

### Metrics on Test Set

```python
# Test metrics (final report)
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.1%}")
```

**Purpose:**
- Final model evaluation
- Report to stakeholders
- Compare with literature
- **Use only once!**

## Next Steps

After data splitting:

1. **[Model Selection](05_model_selection.md)** - Choose appropriate algorithms
2. **[Model Training](06_model_training.md)** - Train models with cross-validation
3. **[Model Evaluation](07_model_evaluation.md)** - Comprehensive evaluation on test set

## Code Reference

Full implementation: `src/data_splitting.py`

Key functions:
- `split_data()` - Stratified train/test split
- `create_cross_validation_folds()` - K-fold CV splitter
- `get_split_info()` - Split statistics
- `validate_split()` - Split quality checks
- `get_cv_split_info()` - CV fold information

See [API Reference](08_api_reference.md) for complete function signatures.
