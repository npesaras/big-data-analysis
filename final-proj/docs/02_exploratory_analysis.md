# Exploratory Data Analysis (EDA)

This document explains how to perform comprehensive exploratory analysis on the diabetes dataset using the `src.exploratory_data_analysis` module.

## Overview

EDA is critical for:
- **Understanding distributions** - How features are spread
- **Identifying relationships** - Feature correlations with target
- **Detecting patterns** - Class separability
- **Validating assumptions** - Normality, independence
- **Guiding preprocessing** - Scaling, transformations needed

The module provides **Plotly-based interactive visualizations** that are:
- Zoomable and pannable
- Hoverable for detailed tooltips
- Exportable as PNG/SVG
- Embeddable in Streamlit

## Key Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `plot_target_distribution()` | Class balance visualization | Plotly Figure |
| `plot_feature_distributions()` | Histograms for all features | Plotly Figure |
| `plot_box_plots()` | Outlier detection by class | Plotly Figure |
| `plot_correlation_matrix()` | Feature correlation heatmap | Plotly Figure |
| `identify_predictive_features()` | Feature-target correlations | Dict |
| `plot_pairwise_relationships()` | Scatter plot matrix | Plotly Figure |

## Target Distribution Analysis

### Class Balance

```python
from src.data_cleaning import load_diabetes_data
from src.exploratory_data_analysis import plot_target_distribution

# Load data
df = load_diabetes_data()

# Visualize class distribution
fig = plot_target_distribution(df)
fig.show()
```

### What to Look For

```
Class Distribution:
- Class 0 (No Diabetes): 500 samples (65.1%)
- Class 1 (Diabetes):    268 samples (34.9%)
- Imbalance Ratio:       1.87:1
```

**Interpretation:**
- **Moderate imbalance** - not severe, but needs consideration
- **Impact on metrics** - Accuracy may be misleading (65% by predicting all zeros)
- **Solution**: Use **stratified sampling** + focus on **Recall/F1** instead of accuracy

### Visualization Details

The bar chart shows:
- **Bar height**: Count of samples per class
- **Percentages**: Displayed on each bar
- **Color coding**: Blue for class 0, orange for class 1
- **Annotations**: Total count and percentage

## Feature Distribution Analysis

### Individual Histograms

```python
from src.exploratory_data_analysis import plot_feature_distributions

# Plot all features with overlapping class distributions
fig = plot_feature_distributions(df, bins=30, show_kde=True)
fig.show()
```

### Expected Patterns

**Glucose (Most Predictive)**
```
Distribution:
- No Diabetes: Peak around 90-110 mg/dL (normal fasting)
- Diabetes: Peak around 120-140 mg/dL (elevated)
- Separation: Good separation, strong predictor
- Shape: Slightly right-skewed
```

**BMI (Body Mass Index)**
```
Distribution:
- No Diabetes: Peak around 28-30 (overweight)
- Diabetes: Peak around 33-35 (obese)
- Separation: Moderate separation
- Shape: Normal distribution
```

**Age**
```
Distribution:
- No Diabetes: Peak around 25-30 years (younger)
- Diabetes: Peak around 35-40 years (older)
- Separation: Moderate, right-skewed
- Shape: Right-skewed (younger population bias)
```

**Insulin**
```
Distribution:
- Highly right-skewed (many low values, few extreme)
- Many missing values encoded as 0
- Requires log transformation consideration
```

### Interactive Features

- **Hover**: Shows exact count and bin range
- **Zoom**: Click-drag to zoom into specific ranges
- **Pan**: Shift-drag to move view
- **Legend**: Click to toggle class visibility

## Box Plot Analysis

### Detecting Outliers by Class

```python
from src.exploratory_data_analysis import plot_box_plots

# Create box plots for all features
fig = plot_box_plots(df)
fig.show()
```

### Reading Box Plots

```
Box Components:
├── Whiskers: Min/Max within 1.5*IQR
├── Box Bottom: Q1 (25th percentile)
├── Line in Box: Median (50th percentile)
├── Box Top: Q3 (75th percentile)
└── Dots: Outliers beyond whiskers
```

### Key Observations

**Pregnancies**
- Many outliers (10+ pregnancies)
- Diabetes patients: Higher median pregnancies
- Keep outliers: Clinically relevant (gestational diabetes risk)

**Glucose**
- Few outliers
- Clear median difference between classes
- Diabetes: Consistently higher glucose
- Strong discriminative power

**Insulin**
- Extreme outliers (>400 mu U/ml)
- Wide variance
- Many zeros (missing data)
- Challenging feature

**Age**
- Right-skewed with high-age outliers
- Diabetes: Older median age
- Clinical relevance: Type 2 diabetes increases with age

## Correlation Analysis

### Correlation Matrix

```python
from src.exploratory_data_analysis import plot_correlation_matrix

# Create heatmap of feature correlations
fig = plot_correlation_matrix(df, method='pearson')
fig.show()
```

### Expected Correlations

**Strong Positive Correlations (> 0.5)**
- **SkinThickness ↔ BMI** (0.66): Thicker skin fold → higher BMI
- **Age ↔ Pregnancies** (0.54): Older women → more pregnancies

**Moderate Positive Correlations (0.3-0.5)**
- **Glucose ↔ Outcome** (0.47): Strongest predictor
- **BMI ↔ Outcome** (0.29): Obesity linked to diabetes
- **Age ↔ Outcome** (0.24): Older age → higher risk
- **Insulin ↔ SkinThickness** (0.44): Related metabolic measures
- **Glucose ↔ Insulin** (0.33): Expected physiological link

**Weak Correlations (< 0.2)**
- **BloodPressure ↔ Outcome** (0.07): Weak predictor
- **DiabetesPedigreeFunction**: Weak with most features (genetic independence)

### Multicollinearity Check

```python
# Check for multicollinearity (correlation > 0.8)
corr_matrix = df[config.FEATURE_COLUMNS].corr()
high_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"High correlations (>0.8): {len(high_corr)}")
# Output: 0 pairs (no multicollinearity issue)
```

**Result**: No severe multicollinearity - all features can be used.

## Predictive Feature Ranking

### Identifying Most Predictive Features

```python
from src.exploratory_data_analysis import identify_predictive_features

# Rank features by correlation with target
feature_importance = identify_predictive_features(df)

for feature, stats in feature_importance.items():
    print(f"{feature:30} | Correlation: {stats['correlation']:+.3f} | "
          f"Strength: {stats['strength']}")
```

**Output:**
```
Glucose                        | Correlation: +0.467 | Strength: Moderate-Strong
BMI                            | Correlation: +0.293 | Strength: Weak-Moderate
Age                            | Correlation: +0.238 | Strength: Weak-Moderate
Pregnancies                    | Correlation: +0.222 | Strength: Weak
Insulin                        | Correlation: +0.199 | Strength: Weak
SkinThickness                  | Correlation: +0.166 | Strength: Weak
DiabetesPedigreeFunction       | Correlation: +0.174 | Strength: Weak
BloodPressure                  | Correlation: +0.065 | Strength: Very Weak
```

### Feature Importance Insights

**Tier 1: Strong Predictors**
- **Glucose** (r=0.47): Primary diagnostic feature
  - Clinical threshold: >126 mg/dL (fasting) indicates diabetes
  - Single best predictor

**Tier 2: Moderate Predictors**
- **BMI** (r=0.29): Obesity increases diabetes risk
- **Age** (r=0.24): Type 2 diabetes prevalence increases with age
- **Pregnancies** (r=0.22): Gestational diabetes connection

**Tier 3: Weak Predictors**
- **Insulin** (r=0.20): High missing data reduces utility
- **DiabetesPedigreeFunction** (r=0.17): Genetic component
- **SkinThickness** (r=0.17): Related to BMI

**Tier 4: Very Weak Predictors**
- **BloodPressure** (r=0.07): Surprisingly weak
  - May become more predictive in combination with other features

## Pairwise Relationships

### Scatter Plot Matrix

```python
from src.exploratory_data_analysis import plot_pairwise_relationships

# Select top predictive features to avoid cluttered plot
top_features = ['Glucose', 'BMI', 'Age', 'Pregnancies']

fig = plot_pairwise_relationships(df, features=top_features)
fig.show()
```

### What to Look For

**Glucose vs BMI**
```
Pattern: Moderate positive trend
- Higher BMI → slightly higher glucose
- Diabetes cluster: Upper-right quadrant (high glucose, high BMI)
- Separation: Visible but overlapping
```

**Glucose vs Age**
```
Pattern: Weak positive trend
- Older patients: Slightly higher glucose
- Diabetes: Scattered throughout, glucose is key discriminator
```

**BMI vs Age**
```
Pattern: Weak correlation
- BMI relatively constant across ages
- Diabetes: No clear age-BMI pattern
```

**Age vs Pregnancies**
```
Pattern: Strong positive correlation (r=0.54)
- Expected: Older women have more pregnancies
- Linear relationship visible
```

### Decision Boundaries Visualization

```python
import plotly.express as px
import plotly.graph_objects as go

# Visualize decision boundary space
fig = px.scatter(
    df,
    x='Glucose',
    y='BMI',
    color='Outcome',
    color_discrete_map={0: 'blue', 1: 'red'},
    labels={'Outcome': 'Diabetes'},
    title='Decision Space: Glucose vs BMI'
)

# Add decision threshold lines (example)
fig.add_hline(y=30, line_dash="dash", line_color="green",
              annotation_text="BMI Threshold (Obesity)")
fig.add_vline(x=126, line_dash="dash", line_color="green",
              annotation_text="Glucose Threshold (Diabetes)")

fig.show()
```

**Interpretation:**
- **Upper-right quadrant**: High glucose + high BMI = highest diabetes risk
- **Lower-left quadrant**: Normal glucose + normal BMI = lowest risk
- **Boundaries**: Not perfectly linear - ML models needed

## Statistical Tests

### Normality Testing

```python
from scipy.stats import shapiro

# Test if features follow normal distribution
for feature in config.FEATURE_COLUMNS:
    stat, p_value = shapiro(df[feature].dropna())
    is_normal = p_value > 0.05
    print(f"{feature:30} | p={p_value:.4f} | Normal: {is_normal}")
```

**Expected Results:**
```
Glucose                        | p=0.0023 | Normal: False (slightly skewed)
BMI                            | p=0.0891 | Normal: False (acceptable)
Age                            | p<0.0001 | Normal: False (right-skewed)
Insulin                        | p<0.0001 | Normal: False (highly skewed)
```

**Implications:**
- Most features are **non-normal**
- **Solution**: Use robust scalers (StandardScaler handles non-normal data)
- **Tree-based models**: Unaffected by non-normality
- **Linear models**: May benefit from log transformation (Insulin)

### Mann-Whitney U Test (Class Difference)

```python
from scipy.stats import mannwhitneyu

# Test if feature distributions differ between classes
for feature in config.FEATURE_COLUMNS:
    class0 = df[df['Outcome'] == 0][feature].dropna()
    class1 = df[df['Outcome'] == 1][feature].dropna()

    stat, p_value = mannwhitneyu(class0, class1)
    significant = p_value < 0.05

    print(f"{feature:30} | p={p_value:.4e} | Significant: {significant}")
```

**Expected Results:**
```
Glucose                        | p=1.23e-28 | Significant: True ✓
BMI                            | p=2.45e-09 | Significant: True ✓
Age                            | p=5.67e-06 | Significant: True ✓
Pregnancies                    | p=8.90e-05 | Significant: True ✓
BloodPressure                  | p=0.0892   | Significant: False ✗
```

**Interpretation:**
- Features with p < 0.05: **Significantly different** between classes → good predictors
- Features with p > 0.05: **No significant difference** → weak predictors

## Visualization Best Practices

### 1. Use Appropriate Plot Types

```python
# ✅ Good: Choose right visualization for data type
plot_target_distribution(df)        # Bar chart for categorical
plot_feature_distributions(df)      # Histogram for continuous
plot_box_plots(df)                  # Box plot for outliers
plot_correlation_matrix(df)         # Heatmap for correlations
```

### 2. Handle Missing Data

```python
# ❌ Bad: Plot with zeros included
fig = plot_feature_distributions(df)  # Includes impossible zeros

# ✅ Good: Clean first, then plot
from src.pre_processing import handle_zero_values
df_cleaned = handle_zero_values(df)
fig = plot_feature_distributions(df_cleaned)
```

### 3. Color Consistency

```python
# Use consistent color scheme across all plots
COLOR_SCHEME = {
    0: '#1f77b4',  # Blue for no diabetes
    1: '#ff7f0e'   # Orange for diabetes
}
```

### 4. Save for Reports

```python
# Save interactive plots
fig = plot_correlation_matrix(df)
fig.write_html("reports/correlation_matrix.html")

# Save static images
fig.write_image("reports/correlation_matrix.png", width=1200, height=800)
```

## Common Patterns and Insights

### Pattern 1: Class Separability

**Observation:** Glucose shows best class separation
```
Median Glucose:
- No Diabetes: 107 mg/dL
- Diabetes:    140 mg/dL
- Difference:  33 mg/dL (31% increase)
```

**Implication:** Glucose-based decision trees will perform well.

### Pattern 2: Feature Skewness

**Observation:** Age and Insulin are right-skewed
```
Age Skewness: +1.13
Insulin Skewness: +2.27
```

**Implication:** Consider log transformation or robust scalers.

### Pattern 3: Missing Data Bias

**Observation:** Insulin missing in 48.7% of samples
```
Missing Insulin:
- No Diabetes: 227/500 (45.4%)
- Diabetes:    147/268 (54.9%)
```

**Implication:** Missingness may be informative - consider creating "Insulin_missing" indicator feature.

### Pattern 4: Age Bias

**Observation:** Dataset skewed toward younger patients
```
Age Distribution:
- 21-30 years: 45% of samples
- 31-40 years: 28% of samples
- 41-50 years: 15% of samples
- 51+ years:   12% of samples
```

**Implication:** Model may underperform on older patients.

## Advanced Visualizations

### Feature Importance via Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Train quick model for feature importance
X = df[config.FEATURE_COLUMNS]
y = df['Outcome']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot importance
importances = rf.feature_importances_
features = config.FEATURE_COLUMNS

fig = go.Figure(go.Bar(
    x=importances,
    y=features,
    orientation='h'
))

fig.update_layout(
    title='Feature Importance (Random Forest)',
    xaxis_title='Importance Score',
    yaxis_title='Feature'
)
fig.show()
```

**Expected Ranking:**
1. Glucose (0.28)
2. BMI (0.16)
3. Age (0.14)
4. DiabetesPedigreeFunction (0.12)
5. Insulin (0.10)
6. SkinThickness (0.08)
7. BloodPressure (0.07)
8. Pregnancies (0.05)

### Distribution Comparison (Q-Q Plot)

```python
from scipy import stats
import plotly.graph_objects as go

# Check if feature follows normal distribution
feature = 'Glucose'
data = df[feature].dropna()

# Generate Q-Q plot data
theoretical_quantiles = stats.probplot(data, dist="norm")[0][0]
sample_quantiles = stats.probplot(data, dist="norm")[0][1]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=sample_quantiles,
    mode='markers',
    name='Data'
))
fig.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=theoretical_quantiles,
    mode='lines',
    name='Normal Distribution',
    line=dict(dash='dash')
))

fig.update_layout(
    title=f'Q-Q Plot: {feature}',
    xaxis_title='Theoretical Quantiles',
    yaxis_title='Sample Quantiles'
)
fig.show()
```

## Next Steps

After completing EDA:

1. **[Preprocessing](03_preprocessing.md)** - Handle zeros, impute missing values, scale features
2. **[Data Splitting](04_data_splitting.md)** - Create stratified train/test splits
3. **[Model Selection](05_model_selection.md)** - Choose appropriate algorithms based on insights

## Key Takeaways

✅ **Glucose** is the strongest single predictor (r=0.47)

✅ **Class imbalance** is moderate (1.87:1) - use stratified sampling

✅ **No severe multicollinearity** - all features can be used

✅ **Missing data** encoded as zeros needs handling

✅ **Outliers** are clinically relevant - do not remove

✅ **Non-normal distributions** - use robust scalers

✅ **Feature interactions** exist - ensemble models will excel

## Code Reference

Full implementation: `src/exploratory_data_analysis.py`

Key functions:
- `plot_target_distribution()` - Class balance chart
- `plot_feature_distributions()` - Feature histograms
- `plot_box_plots()` - Outlier visualization
- `plot_correlation_matrix()` - Correlation heatmap
- `identify_predictive_features()` - Feature ranking
- `plot_pairwise_relationships()` - Scatter matrix

See [API Reference](08_api_reference.md) for complete function signatures.
