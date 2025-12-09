# Model Training

This document explains the model training procedures in `src.model_training`, including single model training, cross-validation, and batch training of all models.

## Overview

The training module provides:
- **Single model training** - Train individual model on train set
- **Cross-validation training** - K-fold CV for robust evaluation
- **Batch training** - Train all 9 models simultaneously
- **Performance tracking** - Training time and CV scores
- **Final model training** - Retraining on full dataset

## Training Pipeline

```
Preprocessed Data
    ↓
Split into Train/Test
    ↓
Cross-Validation on Train Set (10-fold)
    ├─ Fold 1 → Score 1
    ├─ Fold 2 → Score 2
    ├─ ...
    └─ Fold 10 → Score 10
    ↓
Average CV Score (with std)
    ↓
Select Best Model
    ↓
Retrain on Full Training Set
    ↓
Evaluate on Test Set (once!)
```

## Single Model Training

### Basic Training

```python
from src.model_selection import create_single_model
from src.model_training import train_single_model

# Create model
model = create_single_model('Random Forest')

# Train on training data
trained_model, training_time = train_single_model(
    model,
    X_train,
    y_train
)

print(f"Training completed in {training_time:.2f} seconds")
print(f"Training accuracy: {trained_model.score(X_train, y_train):.1%}")
```

### Function Signature

```python
def train_single_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Any, float]:
    """
    Train a single model and measure training time.

    Parameters
    ----------
    model : sklearn estimator
        Untrained model instance
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels

    Returns
    -------
    model : sklearn estimator
        Trained model
    training_time : float
        Time taken to train (seconds)
    """
```

### Training Examples for Each Model

**Fast Models (<0.1s):**
```python
# Naive Bayes - instant training
nb_model = create_single_model('Gaussian Naive Bayes')
trained_nb, time = train_single_model(nb_model, X_train, y_train)
# time ≈ 0.01s

# Logistic Regression - very fast
lr_model = create_single_model('Logistic Regression')
trained_lr, time = train_single_model(lr_model, X_train, y_train)
# time ≈ 0.03s
```

**Medium Speed Models (0.1-1s):**
```python
# Decision Tree
dt_model = create_single_model('Decision Tree')
trained_dt, time = train_single_model(dt_model, X_train, y_train)
# time ≈ 0.05s

# KNN (no training, just stores data)
knn_model = create_single_model('K-Nearest Neighbors')
trained_knn, time = train_single_model(knn_model, X_train, y_train)
# time ≈ 0.01s
```

**Slow Models (>1s):**
```python
# Random Forest - 100 trees
rf_model = create_single_model('Random Forest')
trained_rf, time = train_single_model(rf_model, X_train, y_train)
# time ≈ 1-2s

# SVM with RBF kernel
svm_model = create_single_model('Support Vector Machine')
trained_svm, time = train_single_model(svm_model, X_train, y_train)
# time ≈ 2-5s

# MLP Neural Network
mlp_model = create_single_model('MLP Neural Network')
trained_mlp, time = train_single_model(mlp_model, X_train, y_train)
# time ≈ 3-10s
```

## Cross-Validation Training

### Why Cross-Validation?

**Problem with single split:**
```python
# Single train/test split
X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.1%}")  # 78% - but is this reliable?

# Different random seed
X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=999)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.1%}")  # 73% - very different!
```

**Solution: Cross-Validation**
```python
# 10-fold CV - more robust estimate
cv_results = train_with_cross_validation(model, X, y, n_folds=10)
print(f"Mean Accuracy: {cv_results['mean_score']:.1%} "
      f"(+/- {cv_results['std_score']*2:.1%})")
# Output: Mean Accuracy: 75.5% (+/- 3.2%)
```

### Training with Cross-Validation

```python
from src.model_training import train_with_cross_validation
from src.data_splitting import create_cross_validation_folds

# Create CV splitter
cv = create_cross_validation_folds(df, n_folds=10)

# Train with CV
model = create_single_model('Random Forest')
cv_results = train_with_cross_validation(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring='recall'  # Focus on recall for medical diagnosis
)

print("Cross-Validation Results:")
print(f"  Mean Score: {cv_results['mean_score']:.3f}")
print(f"  Std Score:  {cv_results['std_score']:.3f}")
print(f"  All Scores: {cv_results['scores']}")
print(f"  Training Time: {cv_results['training_time']:.2f}s")
```

**Output:**
```
Cross-Validation Results:
  Mean Score: 0.756
  Std Score:  0.042
  All Scores: [0.78, 0.75, 0.79, 0.77, 0.76, 0.78, 0.77, 0.79, 0.76, 0.75]
  Training Time: 1.23s
```

### Function Signature

```python
def train_with_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[Any] = None,
    n_folds: int = 10,
    scoring: str = 'accuracy',
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Train model using k-fold cross-validation.

    Parameters
    ----------
    model : sklearn estimator
        Model to train
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    cv : cross-validation generator, optional
        Custom CV splitter. If None, creates StratifiedKFold
    n_folds : int, default=10
        Number of folds (used if cv=None)
    scoring : str, default='accuracy'
        Scoring metric: 'accuracy', 'recall', 'precision', 'f1', 'roc_auc'
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all CPU cores)

    Returns
    -------
    dict
        - 'mean_score': Mean CV score
        - 'std_score': Standard deviation of scores
        - 'scores': Array of individual fold scores
        - 'training_time': Total time for all folds
        - 'n_folds': Number of folds used
    """
```

### Choosing Scoring Metric

**For medical diagnosis (diabetes):**

| Metric | Formula | Use Case | Our Choice |
|--------|---------|----------|------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes | No |
| **Recall** | TP/(TP+FN) | **Minimize false negatives** | ✅ Yes |
| **Precision** | TP/(TP+FP) | Minimize false positives | No |
| **F1 Score** | 2·(Precision·Recall)/(Precision+Recall) | Balance precision & recall | Maybe |
| **ROC AUC** | Area under ROC curve | Overall discrimination | Yes |

**Why Recall?**
- False Negative (predict no diabetes, but has diabetes) = **dangerous**
- False Positive (predict diabetes, but doesn't have) = extra tests, but safer
- **Medical priority: Don't miss diabetic patients**

```python
# Train with recall focus
cv_results = train_with_cross_validation(
    model, X_train, y_train,
    scoring='recall'  # Prioritize catching diabetic patients
)
```

## Batch Training All Models

### Training All 9 Models

```python
from src.model_training import train_all_models

# Train all models with CV
results = train_all_models(
    X_train,
    y_train,
    cv_folds=10,
    scoring='recall'
)

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']*2:.3f})")
    print(f"  Training Time: {metrics['training_time']:.2f}s")
```

**Output:**
```
Logistic Regression:
  CV Score: 0.741 (+/- 0.048)
  Training Time: 0.15s

Decision Tree:
  CV Score: 0.698 (+/- 0.062)
  Training Time: 0.08s

Random Forest:
  CV Score: 0.781 (+/- 0.041)
  Training Time: 2.34s

Gaussian Naive Bayes:
  CV Score: 0.745 (+/- 0.051)
  Training Time: 0.02s

K-Nearest Neighbors:
  CV Score: 0.712 (+/- 0.054)
  Training Time: 0.03s

Support Vector Machine:
  CV Score: 0.738 (+/- 0.047)
  Training Time: 4.12s

AdaBoost:
  CV Score: 0.726 (+/- 0.056)
  Training Time: 0.67s

Perceptron:
  CV Score: 0.685 (+/- 0.068)
  Training Time: 0.04s

MLP Neural Network:
  CV Score: 0.752 (+/- 0.049)
  Training Time: 8.91s
```

### Function Signature

```python
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 10,
    scoring: str = 'recall',
    n_jobs: int = -1
) -> Dict[str, Dict[str, float]]:
    """
    Train all models with cross-validation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv_folds : int, default=10
        Number of CV folds
    scoring : str, default='recall'
        Scoring metric
    n_jobs : int, default=-1
        Parallel jobs (-1 = all cores)

    Returns
    -------
    dict
        Dictionary mapping model names to results:
        - 'cv_mean': Mean CV score
        - 'cv_std': Std deviation of CV scores
        - 'cv_scores': Individual fold scores
        - 'training_time': Total training time
        - 'trained_model': Final trained model on full training set
    """
```

### Comparing Results

```python
from src.model_training import compare_cv_results

# Train all models
results = train_all_models(X_train, y_train)

# Compare and rank
comparison = compare_cv_results(results)

print("Model Ranking (by CV Score):")
for rank, (model_name, score, std) in enumerate(comparison, 1):
    print(f"{rank}. {model_name:30} {score:.3f} (+/- {std*2:.3f})")
```

**Output:**
```
Model Ranking (by CV Score):
1. Random Forest                  0.781 (+/- 0.041)  ⭐ Best
2. MLP Neural Network             0.752 (+/- 0.049)
3. Gaussian Naive Bayes           0.745 (+/- 0.051)
4. Logistic Regression            0.741 (+/- 0.048)
5. Support Vector Machine         0.738 (+/- 0.047)
6. AdaBoost                       0.726 (+/- 0.056)
7. K-Nearest Neighbors            0.712 (+/- 0.054)
8. Decision Tree                  0.698 (+/- 0.062)
9. Perceptron                     0.685 (+/- 0.068)
```

## Final Model Training

### Retraining on Full Dataset

After selecting best model via CV, retrain on **entire training set** for maximum performance:

```python
from src.model_training import train_final_model

# Best model from CV
best_model_name = 'Random Forest'
best_model = create_single_model(best_model_name)

# Train on full training set (no CV, no split)
final_model, training_time = train_final_model(
    best_model,
    X_train,
    y_train
)

print(f"Final model trained in {training_time:.2f}s")
print(f"Training set accuracy: {final_model.score(X_train, y_train):.1%}")

# Now evaluate ONCE on test set
test_accuracy = final_model.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy:.1%}")
```

**Why retrain?**
- CV uses 90% of data per fold (10-fold CV)
- Final model uses 100% of training data
- More data = better model
- Test set is still unseen (no leakage)

### Saving Trained Models

```python
from src.utils import save_model

# Save best model
save_model(final_model, f"{config.MODELS_DIR}/best_model.joblib")

# Save all trained models
results = train_all_models(X_train, y_train)
for model_name, metrics in results.items():
    model = metrics['trained_model']
    filename = f"{config.MODELS_DIR}/{model_name.replace(' ', '_').lower()}.joblib"
    save_model(model, filename)
```

## Training Best Practices

### 1. Always Use Cross-Validation for Model Selection

```python
# ✅ Correct: CV for selection, retrain for deployment
cv_results = train_all_models(X_train, y_train, cv_folds=10)
best_model_name = max(cv_results, key=lambda k: cv_results[k]['cv_mean'])
best_model = create_single_model(best_model_name)
final_model, _ = train_final_model(best_model, X_train, y_train)

# ❌ Wrong: Select based on single train/test split
for model_name, model in create_all_models().items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Using test set for selection!
```

### 2. Use Stratified Folds

```python
# ✅ Maintains class balance in each fold
cv = create_cross_validation_folds(df, n_folds=10, stratified=True)

# ❌ May create imbalanced folds
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True, random_state=42)
```

### 3. Choose Appropriate Scoring Metric

```python
# For medical diagnosis (diabetes)
scoring = 'recall'  # ✅ Minimize false negatives

# For balanced classes
scoring = 'accuracy'  # ✅ Overall correctness

# For fraud detection
scoring = 'precision'  # ✅ Minimize false positives
```

### 4. Monitor Training Time

```python
import time

# Track time for each model
training_times = {}
for model_name, model in create_all_models().items():
    start = time.time()
    model.fit(X_train, y_train)
    training_times[model_name] = time.time() - start

# Identify fast models
fast_models = {k: v for k, v in training_times.items() if v < 0.1}
print(f"Fast models (<0.1s): {list(fast_models.keys())}")
```

### 5. Parallelize When Possible

```python
# Use n_jobs=-1 to use all CPU cores
cv_results = train_with_cross_validation(
    model, X_train, y_train,
    n_jobs=-1  # Parallel processing
)

# Some models support parallel training
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
```

## Common Training Issues

### Issue 1: Model Not Converging

```python
# ❌ Problem: ConvergenceWarning for Logistic Regression
model = LogisticRegression(max_iter=100)  # Too few iterations
model.fit(X_train, y_train)

# ✅ Solution: Increase max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Issue 2: Training Too Slow

```python
# ❌ Problem: SVM taking 10+ minutes
svm = SVC(kernel='rbf', C=10.0)
svm.fit(X_train, y_train)  # Very slow

# ✅ Solution 1: Use linear kernel
svm = SVC(kernel='linear')

# ✅ Solution 2: Reduce sample size (for initial experiments)
X_train_sample = X_train.sample(n=100, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
svm.fit(X_train_sample, y_train_sample)

# ✅ Solution 3: Use LinearSVC (faster)
from sklearn.svm import LinearSVC
svm = LinearSVC()
```

### Issue 3: Overfitting

```python
# Signs of overfitting
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)  # 95%
test_score = model.score(X_test, y_test)    # 70%  ← Big gap!

# ✅ Solutions:
# 1. Regularization
model = RandomForestClassifier(max_depth=5, min_samples_leaf=10)

# 2. More data (use CV to check)
# 3. Feature selection
# 4. Simpler model
```

### Issue 4: Class Imbalance

```python
# ❌ Problem: Model predicts all class 0
print(y_train.value_counts())
# 0: 400
# 1: 214  (35% minority class)

# ✅ Solution: Use class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

## Monitoring Training Progress

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='recall'
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Recall Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

# Test different hyperparameters
param_range = [1, 5, 10, 20, 50, 100]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range,
    cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label='Training')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Next Steps

After training:

1. **[Model Evaluation](07_model_evaluation.md)** - Comprehensive evaluation on test set
2. **Hyperparameter Tuning** - Optimize best model
3. **Model Deployment** - Save and deploy best model

## Code Reference

Full implementation: `src/model_training.py`

Key functions:
- `train_single_model()` - Train one model
- `train_with_cross_validation()` - K-fold CV training
- `train_all_models()` - Batch train all models
- `compare_cv_results()` - Rank models
- `train_final_model()` - Retrain on full training set

See [API Reference](08_api_reference.md) for complete function signatures.
