# Model Selection

This document explains the 9 classification algorithms implemented in `src.model_selection` and provides guidance on choosing the right model for diabetes prediction.

## Overview

The system implements **9 diverse classification algorithms** to compare performance and find the optimal model. This multi-model approach ensures:
- **Algorithm comparison** - Identify best performer for this specific problem
- **Ensemble potential** - Combine predictions from multiple models
- **Robustness** - Different algorithms capture different patterns
- **Learning** - Understand which approaches work best for medical data

## Available Algorithms

| Algorithm | Type | Best For | Interpretability | Training Speed |
|-----------|------|----------|------------------|----------------|
| **Logistic Regression** | Linear | Linear relationships | ★★★★★ | Fast |
| **Decision Tree** | Tree | Non-linear, interpretable | ★★★★☆ | Fast |
| **Random Forest** | Ensemble (Bagging) | General purpose | ★★★☆☆ | Medium |
| **Gaussian Naive Bayes** | Probabilistic | Independent features | ★★★★☆ | Very Fast |
| **K-Nearest Neighbors** | Instance-based | Local patterns | ★★☆☆☆ | Fast (Slow prediction) |
| **Support Vector Machine** | Kernel | Complex boundaries | ★★☆☆☆ | Slow |
| **AdaBoost** | Ensemble (Boosting) | Hard examples | ★★★☆☆ | Medium |
| **Perceptron** | Linear | Simple binary | ★★★★☆ | Very Fast |
| **MLP Neural Network** | Neural Network | Complex patterns | ★☆☆☆☆ | Medium-Slow |

## Creating All Models

### Basic Usage

```python
from src.model_selection import create_all_models

# Create dictionary of all 9 models with default config
models = create_all_models()

print(f"Number of models: {len(models)}")
print("Available models:")
for name in models.keys():
    print(f"  - {name}")
```

**Output:**
```
Number of models: 9
Available models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gaussian Naive Bayes
  - K-Nearest Neighbors
  - Support Vector Machine
  - AdaBoost
  - Perceptron
  - MLP Neural Network
```

### Function Signature

```python
def create_all_models() -> Dict[str, Any]:
    """
    Create all classification models with configured hyperparameters.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to sklearn estimator instances

    Note
    ----
    All hyperparameters are defined in config.MODELS_CONFIG.
    Each model is instantiated with config.RANDOM_STATE for reproducibility.
    """
```

## Model Details and Configuration

### 1. Logistic Regression

**Theory:**
- Learns linear decision boundary: $P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$
- Outputs probability of class 1
- Maximum likelihood estimation via gradient descent

**Configuration:**
```python
{
    'model': LogisticRegression,
    'params': {
        'max_iter': 1000,           # Iterations for convergence
        'solver': 'liblinear',      # Optimization algorithm
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `max_iter=1000`: Sufficient for convergence on scaled data
- `solver='liblinear'`: Good for small datasets, handles L1/L2 regularization
- Default C=1.0 (no strong regularization)

**Strengths:**
- Very interpretable (feature coefficients)
- Fast training and prediction
- Probabilistic outputs
- Works well with scaled features

**Weaknesses:**
- Assumes linear relationship
- Cannot model complex interactions without feature engineering
- Sensitive to feature scale (requires scaling)

**Best For:**
- Baseline model
- When interpretability is crucial
- Linear relationships

**Expected Performance:**
- Accuracy: ~76-78%
- Works well with Glucose (strong linear predictor)

### 2. Decision Tree

**Theory:**
- Learns hierarchical if-then rules
- Splits based on information gain or Gini impurity
- Creates axis-aligned decision boundaries

**Configuration:**
```python
{
    'model': DecisionTreeClassifier,
    'params': {
        'max_depth': 5,             # Maximum tree depth
        'min_samples_split': 20,    # Minimum samples to split
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `max_depth=5`: Prevents overfitting (trees >10 overfit easily)
- `min_samples_split=20`: Requires at least 20 samples to create split
- Unpruned tree would have depth ~15 and overfit

**Strengths:**
- Highly interpretable (visualizable tree)
- Handles non-linear relationships
- No need for feature scaling
- Fast training and prediction
- Handles missing values naturally

**Weaknesses:**
- Prone to overfitting
- High variance (small data change → different tree)
- Axis-aligned splits (struggles with diagonal boundaries)

**Best For:**
- Exploratory analysis
- Understanding feature interactions
- When interpretability is crucial

**Expected Performance:**
- Accuracy: ~72-75%
- May overfit without proper tuning

### 3. Random Forest

**Theory:**
- Ensemble of multiple decision trees
- Each tree trained on bootstrap sample
- Final prediction: majority vote
- Reduces variance through averaging

**Configuration:**
```python
{
    'model': RandomForestClassifier,
    'params': {
        'n_estimators': 100,        # Number of trees
        'max_depth': 10,            # Depth of each tree
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `n_estimators=100`: 100 trees balances performance and computation
- `max_depth=10`: Deeper than single tree (ensemble reduces overfitting)
- Default: Uses bootstrap sampling and feature subsampling

**Strengths:**
- Excellent general-purpose algorithm
- Reduces overfitting vs single tree
- Feature importance rankings
- Handles non-linear relationships
- Robust to outliers and noise

**Weaknesses:**
- Less interpretable than single tree
- Slower training and prediction
- Larger memory footprint

**Best For:**
- **Default choice for tabular data**
- When accuracy is priority
- Feature importance analysis

**Expected Performance:**
- Accuracy: ~77-80%
- **Often best performer**

### 4. Gaussian Naive Bayes

**Theory:**
- Applies Bayes' theorem with "naive" independence assumption
- $P(y|x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$
- Assumes features follow Gaussian distribution per class

**Configuration:**
```python
{
    'model': GaussianNB,
    'params': {}  # No hyperparameters to tune
}
```

**Hyperparameters:**
- None! (Simplest model)
- Automatically estimates mean and variance per class

**Strengths:**
- Extremely fast training and prediction
- Works well with small datasets
- Probabilistic predictions
- Handles high-dimensional data
- Few assumptions

**Weaknesses:**
- Assumes feature independence (violated in diabetes data)
- Assumes Gaussian distribution (may not hold)
- Cannot model feature interactions

**Best For:**
- Baseline model
- Fast prototyping
- When features are truly independent

**Expected Performance:**
- Accuracy: ~75-76%
- Surprisingly competitive despite violated assumptions

### 5. K-Nearest Neighbors (KNN)

**Theory:**
- Instance-based learning (lazy learner)
- Classifies based on majority vote of K nearest neighbors
- Distance metric: Euclidean distance
- No training phase (stores entire dataset)

**Configuration:**
```python
{
    'model': KNeighborsClassifier,
    'params': {
        'n_neighbors': 5,           # Number of neighbors
        'weights': 'uniform',       # Equal weight to all neighbors
    }
}
```

**Hyperparameters:**
- `n_neighbors=5`: Standard choice (odd number avoids ties)
- `weights='uniform'`: All neighbors contribute equally
- Alternative: `weights='distance'` (closer neighbors have more influence)

**Strengths:**
- Simple and intuitive
- No training phase
- Non-parametric (no assumptions about data distribution)
- Can model complex boundaries

**Weaknesses:**
- **Requires feature scaling** (critical!)
- Slow prediction (must compute distances to all training samples)
- Memory intensive (stores entire dataset)
- Sensitive to irrelevant features
- Curse of dimensionality

**Best For:**
- Small datasets
- When decision boundary is irregular
- Low-dimensional problems

**Expected Performance:**
- Accuracy: ~74-76%
- Highly dependent on K value and scaling

### 6. Support Vector Machine (SVM)

**Theory:**
- Finds optimal hyperplane that maximizes margin between classes
- Uses kernel trick for non-linear boundaries
- Solves: $\min \frac{1}{2}||w||^2$ subject to $y_i(w \cdot x_i + b) \geq 1$

**Configuration:**
```python
{
    'model': SVC,
    'params': {
        'kernel': 'rbf',            # Radial Basis Function kernel
        'C': 1.0,                   # Regularization parameter
        'probability': True,        # Enable probability estimates
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `kernel='rbf'`: Gaussian kernel allows non-linear boundaries
- `C=1.0`: Regularization strength (balance margin vs errors)
- `probability=True`: Required for predict_proba() (slower training)

**Strengths:**
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile (different kernels)
- Good for non-linear boundaries

**Weaknesses:**
- Slow training (O(n²) to O(n³))
- **Requires feature scaling** (critical!)
- Difficult to interpret
- Many hyperparameters to tune
- Sensitive to kernel choice

**Best For:**
- High-dimensional data
- When data is not linearly separable
- Clear margin of separation exists

**Expected Performance:**
- Accuracy: ~76-78%
- Similar to Logistic Regression but slower

### 7. AdaBoost

**Theory:**
- Adaptive Boosting: Sequential ensemble
- Each weak learner focuses on misclassified examples
- Weighted voting based on learner accuracy
- Final prediction: $H(x) = \text{sign}(\sum_{t=1}^{T} \alpha_t h_t(x))$

**Configuration:**
```python
{
    'model': AdaBoostClassifier,
    'params': {
        'n_estimators': 50,         # Number of weak learners
        'learning_rate': 1.0,       # Shrinkage parameter
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `n_estimators=50`: Number of boosting iterations
- `learning_rate=1.0`: No shrinkage (full weight updates)
- Base estimator: Decision stump (depth=1 tree) by default

**Strengths:**
- Focuses on hard-to-classify examples
- Less prone to overfitting than single tree
- No feature scaling required
- Good with weak learners

**Weaknesses:**
- Sensitive to noisy data and outliers
- Can overfit if too many estimators
- Slower than single tree
- Sequential (cannot parallelize)

**Best For:**
- Imbalanced datasets
- When you have weak learners
- Boosting performance of simple models

**Expected Performance:**
- Accuracy: ~75-77%
- Improves over single decision tree

### 8. Perceptron

**Theory:**
- Simplest neural network (single layer)
- Linear classifier: $\hat{y} = \text{sign}(w \cdot x + b)$
- Online learning algorithm
- Updates: $w = w + \alpha \cdot (y - \hat{y}) \cdot x$

**Configuration:**
```python
{
    'model': Perceptron,
    'params': {
        'max_iter': 1000,           # Maximum passes over data
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `max_iter=1000`: Sufficient for convergence
- Default learning rate: 0.001

**Strengths:**
- Very fast training
- Online learning (can update with new data)
- Simple and interpretable
- Low memory footprint

**Weaknesses:**
- Only learns linearly separable patterns
- Sensitive to feature scale
- Less robust than Logistic Regression
- Can fail to converge if data not linearly separable

**Best For:**
- Large-scale online learning
- Simple baseline
- Linearly separable data

**Expected Performance:**
- Accuracy: ~70-73%
- Underperforms compared to Logistic Regression

### 9. MLP Neural Network

**Theory:**
- Multi-Layer Perceptron: Feedforward neural network
- Hidden layers with non-linear activation (ReLU)
- Backpropagation training
- Can model complex non-linear relationships

**Configuration:**
```python
{
    'model': MLPClassifier,
    'params': {
        'hidden_layer_sizes': (100, 50),  # 2 hidden layers (100, 50 neurons)
        'max_iter': 1000,                 # Training iterations
        'random_state': 42
    }
}
```

**Hyperparameters:**
- `hidden_layer_sizes=(100, 50)`: Architecture with 2 hidden layers
- `max_iter=1000`: Maximum epochs
- Default: Adam optimizer, ReLU activation

**Architecture:**
```
Input Layer (8 features)
    ↓
Hidden Layer 1 (100 neurons, ReLU)
    ↓
Hidden Layer 2 (50 neurons, ReLU)
    ↓
Output Layer (1 neuron, Logistic)
```

**Strengths:**
- Can model very complex non-linear relationships
- Automatic feature interactions
- Flexible architecture
- Good for large datasets

**Weaknesses:**
- **Requires careful tuning** (architecture, learning rate, regularization)
- Prone to overfitting on small datasets
- Slow training
- Black box (not interpretable)
- **Requires feature scaling** (critical!)

**Best For:**
- Large datasets (>10,000 samples)
- Complex non-linear patterns
- When interpretability is not needed

**Expected Performance:**
- Accuracy: ~75-78%
- May overfit on this small dataset (768 samples)
- Requires more data to outperform Random Forest

## Model Comparison Summary

### Accuracy Ranking (Expected)

1. **Random Forest**: 77-80% ⭐ Best
2. **SVM**: 76-78%
3. **Logistic Regression**: 76-78%
4. **MLP**: 75-78%
5. **Naive Bayes**: 75-76%
6. **AdaBoost**: 75-77%
7. **KNN**: 74-76%
8. **Decision Tree**: 72-75%
9. **Perceptron**: 70-73%

### Training Speed Ranking

1. **Naive Bayes**: <0.01s ⚡ Fastest
2. **Perceptron**: <0.01s
3. **Logistic Regression**: 0.01-0.05s
4. **Decision Tree**: 0.02-0.05s
5. **KNN**: 0.01s (but slow prediction)
6. **AdaBoost**: 0.1-0.5s
7. **Random Forest**: 0.5-2s
8. **MLP**: 1-10s
9. **SVM**: 2-10s 🐢 Slowest

### Interpretability Ranking

1. **Logistic Regression**: Direct feature coefficients
2. **Decision Tree**: Visual tree structure
3. **Naive Bayes**: Probability tables
4. **Perceptron**: Linear weights
5. **Random Forest**: Feature importance (aggregate)
6. **AdaBoost**: Weighted weak learners
7. **KNN**: Instance-based (no model)
8. **SVM**: Support vectors (hard to interpret)
9. **MLP**: Black box

## Choosing the Right Model

### Decision Tree

```
Start
  │
  ├─ Need interpretability? → Logistic Regression or Decision Tree
  │
  ├─ Small dataset (<1000)? → Random Forest or Naive Bayes
  │
  ├─ Large dataset (>10000)? → MLP or SVM
  │
  ├─ Fast prediction needed? → Logistic Regression or Naive Bayes
  │
  ├─ Complex non-linear? → Random Forest or MLP
  │
  └─ Best general choice? → Random Forest ⭐
```

### For Diabetes Prediction

**Recommended:**
1. **Random Forest** - Best balance of accuracy and robustness
2. **Logistic Regression** - Interpretable baseline
3. **SVM** - Alternative for non-linear patterns

**Not Recommended:**
- **MLP** - Overkill for small dataset
- **Perceptron** - Underperforms
- **Decision Tree** - Single tree overfits

## Accessing Model Descriptions

```python
from src.model_selection import get_model_descriptions

# Get descriptions of all models
descriptions = get_model_descriptions()

for name, desc in descriptions.items():
    print(f"\n{name}:")
    print(f"  Type: {desc['type']}")
    print(f"  Description: {desc['description']}")
    print(f"  Pros: {', '.join(desc['pros'])}")
    print(f"  Cons: {', '.join(desc['cons'])}")
```

## Getting Model Hyperparameters

```python
from src.model_selection import get_model_hyperparameters

# Get hyperparameter configuration
hyperparams = get_model_hyperparameters('Random Forest')

print(f"Model: {hyperparams['model_name']}")
print(f"Hyperparameters:")
for param, value in hyperparams['params'].items():
    print(f"  {param}: {value}")
```

**Output:**
```
Model: Random Forest
Hyperparameters:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

## Creating Single Model

```python
from src.model_selection import create_single_model

# Create specific model
rf_model = create_single_model('Random Forest')

# Train it
rf_model.fit(X_train, y_train)

# Predict
predictions = rf_model.predict(X_test)
```

## Model Complexity Analysis

```python
from src.model_selection import get_model_complexity_info

# Analyze model complexity
complexity = get_model_complexity_info('Random Forest')

print(f"Time Complexity: {complexity['time_complexity']}")
print(f"Space Complexity: {complexity['space_complexity']}")
print(f"Parameters to tune: {len(complexity['key_hyperparameters'])}")
```

## Next Steps

After model selection:

1. **[Model Training](06_model_training.md)** - Train all models with cross-validation
2. **[Model Evaluation](07_model_evaluation.md)** - Compare and select best model
3. **Hyperparameter Tuning** - Optimize best model

## Code Reference

Full implementation: `src/model_selection.py`

Key functions:
- `create_all_models()` - Instantiate all 9 models
- `create_single_model()` - Create specific model
- `get_model_descriptions()` - Model information
- `get_model_hyperparameters()` - Get config
- `get_model_complexity_info()` - Complexity analysis

Configuration: `src/config.py` → `MODELS_CONFIG`

See [API Reference](08_api_reference.md) for complete function signatures.
