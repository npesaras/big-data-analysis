"""
ITD105 - Big Data Analytics
Lab Exercise #2 - Part 2: Regression Task
Boston Housing Dataset - Environmental Regression
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATASET
# =============================================================================
print("="*80)
print("REGRESSION TASK - BOSTON HOUSING DATASET")
print("="*80)

# Load the housing dataset
df = pd.read_csv('HousingData.csv')
print(f"\n📊 Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Features: {df.columns.tolist()}")

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\n⚠️  Missing values detected:")
    print(missing_counts[missing_counts > 0])

# =============================================================================
# 2. DEFINE X (FEATURES) AND y (TARGET)
# =============================================================================
X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"\n✅ Features (X): {X.shape}")
print(f"✅ Target (y): {y.shape}")
print(f"   Target variable: MEDV (Median value of homes in $1000s)")
print(f"   Target range: ${y.min():.1f}k - ${y.max():.1f}k")
print(f"   Target mean: ${y.mean():.1f}k")

# =============================================================================
# 3. CREATE PREPROCESSING PIPELINE (IMPUTATION + STANDARDSCALER)
# =============================================================================
print(f"\n🔧 Creating preprocessing pipeline...")

# Pipeline with imputation (for NA values) and scaling
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

print(f"   Pipeline: SimpleImputer (median) → StandardScaler")

# =============================================================================
# 4. MODEL A - TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "="*80)
print("MODEL A: SIMPLE TRAIN-TEST SPLIT")
print("="*80)

# Create full pipeline for Model A
pipeline_a = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# =============================================================================
# 5. FIT MODEL A ON TRAINING DATA
# =============================================================================
print(f"\n🔄 Training Model A...")
pipeline_a.fit(X_train, y_train)
print(f"✅ Model A trained successfully!")

# =============================================================================
# 6. MAKE PREDICTIONS ON TEST SET
# =============================================================================
y_pred_a = pipeline_a.predict(X_test)
print(f"✅ Predictions generated for test set")

# =============================================================================
# 7. CALCULATE METRICS FOR MODEL A
# =============================================================================
mse_a = mean_squared_error(y_test, y_pred_a)
mae_a = mean_absolute_error(y_test, y_pred_a)
r2_a = r2_score(y_test, y_pred_a)
rmse_a = np.sqrt(mse_a)

print(f"\n📈 Performance Metrics (Model A):")
print(f"   Mean Squared Error (MSE): {mse_a:.4f}")
print(f"   Root Mean Squared Error (RMSE): {rmse_a:.4f}")
print(f"   Mean Absolute Error (MAE): {mae_a:.4f}")
print(f"   R-squared (R²): {r2_a:.4f}")

print(f"\n💡 Interpretation:")
print(f"   - On average, predictions are off by ${mae_a:.2f}k")
print(f"   - The model explains {r2_a*100:.2f}% of the variance in home prices")

# =============================================================================
# 8. MODEL B - REPEATED RANDOM TRAIN-TEST SPLITS
# =============================================================================
print("\n" + "="*80)
print("MODEL B: REPEATED RANDOM TRAIN-TEST SPLITS")
print("="*80)

# Create full pipeline for Model B (same as Model A)
pipeline_b = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Use RepeatedKFold for robust evaluation
rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
print(f"\n🔄 Evaluating with Repeated K-Fold:")
print(f"   Number of splits: 10")
print(f"   Number of repeats: 3")
print(f"   Total evaluations: {10 * 3}")
print(f"   This may take a moment...")

# =============================================================================
# 9. CALCULATE SCORES FOR MODEL B
# =============================================================================
# Get cross-validation scores (note: sklearn returns negative MSE and MAE)
scores_mse_b = cross_val_score(
    pipeline_b, X, y, cv=rkf, 
    scoring='neg_mean_squared_error', n_jobs=-1
)
scores_mae_b = cross_val_score(
    pipeline_b, X, y, cv=rkf, 
    scoring='neg_mean_absolute_error', n_jobs=-1
)
scores_r2_b = cross_val_score(
    pipeline_b, X, y, cv=rkf, 
    scoring='r2', n_jobs=-1
)

# Convert to positive values for MSE and MAE
scores_mse_b = -scores_mse_b
scores_mae_b = -scores_mae_b
scores_rmse_b = np.sqrt(scores_mse_b)

# =============================================================================
# 10. PRINT MEAN AND STANDARD DEVIATION FOR MODEL B
# =============================================================================
print(f"\n📈 Performance Metrics (Model B):")
print(f"\n   Mean Squared Error (MSE):")
print(f"      Mean: {scores_mse_b.mean():.4f}")
print(f"      Std Dev: {scores_mse_b.std():.4f}")

print(f"\n   Root Mean Squared Error (RMSE):")
print(f"      Mean: {scores_rmse_b.mean():.4f}")
print(f"      Std Dev: {scores_rmse_b.std():.4f}")

print(f"\n   Mean Absolute Error (MAE):")
print(f"      Mean: {scores_mae_b.mean():.4f}")
print(f"      Std Dev: {scores_mae_b.std():.4f}")

print(f"\n   R-squared (R²):")
print(f"      Mean: {scores_r2_b.mean():.4f}")
print(f"      Std Dev: {scores_r2_b.std():.4f}")

print(f"\n💡 Interpretation:")
print(f"   - Across {len(scores_mae_b)} train-test splits:")
print(f"   - Average prediction error: ${scores_mae_b.mean():.2f}k (±${scores_mae_b.std():.2f}k)")
print(f"   - Model explains {scores_r2_b.mean()*100:.2f}% (±{scores_r2_b.std()*100:.2f}%) of variance")

# =============================================================================
# 11. COMPARISON AND JUSTIFICATION
# =============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON & SELECTION")
print("="*80)

print(f"\n📊 Side-by-Side Comparison:")
print(f"{'Metric':<35} {'Model A (Single Split)':<25} {'Model B (Repeated Splits)':<30}")
print("-" * 90)
print(f"{'Mean Squared Error (MSE)':<35} {mse_a:<25.4f} {scores_mse_b.mean():.4f} (±{scores_mse_b.std():.4f})")
print(f"{'Root Mean Squared Error (RMSE)':<35} {rmse_a:<25.4f} {scores_rmse_b.mean():.4f} (±{scores_rmse_b.std():.4f})")
print(f"{'Mean Absolute Error (MAE)':<35} {mae_a:<25.4f} {scores_mae_b.mean():.4f} (±{scores_mae_b.std():.4f})")
print(f"{'R-squared (R²)':<35} {r2_a:<25.4f} {scores_r2_b.mean():.4f} (±{scores_r2_b.std():.4f})")

print("\n" + "="*80)
print("🏆 SELECTED MODEL: MODEL B (REPEATED RANDOM TRAIN-TEST SPLITS)")
print("="*80)

print("""
📝 JUSTIFICATION:

Model B (Repeated Random Train-Test Splits with 10 splits × 3 repeats) is selected 
as the best approach for the following reasons:

1. 🎯 ROBUST PERFORMANCE ESTIMATION:
   - Model B evaluates performance across 30 different train-test splits
   - Provides mean and standard deviation for all metrics
   - Reduces the impact of a single "lucky" or "unlucky" data split
   - Model A uses only ONE split, which may not be representative

2. 📊 BETTER GENERALIZATION ASSESSMENT:
   - Multiple random splits test the model on diverse subsets of data
   - Reveals model stability and consistency across different data configurations
   - Lower standard deviation indicates more reliable predictions
   - Helps identify if the model is sensitive to specific data partitions

3. 🔬 STATISTICAL CONFIDENCE:
   - Standard deviations provide confidence intervals for performance metrics
   - Enables statistical comparison with other models or benchmarks
   - More scientific and rigorous evaluation methodology
   - Reduces risk of overfitting to a single test set

4. 🏭 REAL-WORLD RELIABILITY:
   - In production, data distribution may vary over time
   - Model B's repeated evaluation simulates this variability
   - Better predictor of how the model will perform on unseen future data
   - Aligns with best practices in machine learning validation

5. ⚖️ MINIMAL ADDITIONAL COST:
   - While Model B requires more computation, the benefit far outweighs the cost
   - Modern computing resources make this approach practical
   - The increased confidence in model performance is worth the extra time

6. 🔄 CONSISTENCY WITH STANDARDS:
   - Repeated cross-validation is recommended in academic and industry settings
   - Aligns with best practices in regression model evaluation
   - Facilitates comparison with other studies and benchmarks

CONCLUSION: Model B (Repeated Random Train-Test Splits) is the optimal choice,
providing robust, reliable, and statistically sound performance estimates that
give us high confidence in the model's ability to generalize to new data.
""")

# =============================================================================
# 12. TRAIN FINAL MODEL ON ENTIRE DATASET & SAVE
# =============================================================================
print("\n" + "="*80)
print("TRAINING FINAL MODEL ON ENTIRE DATASET")
print("="*80)

# Train the pipeline on the entire dataset
pipeline_b.fit(X, y)
print(f"\n✅ Final model trained on all {X.shape[0]} samples")

# Save the model
model_filename = 'regression_model.joblib'
joblib.dump(pipeline_b, model_filename)
print(f"✅ Model saved as: {model_filename}")

# Show example prediction
print(f"\n🔍 Example Prediction:")
sample = X.iloc[0:1]
prediction = pipeline_b.predict(sample)
actual = y.iloc[0]
print(f"   Predicted price: ${prediction[0]:.2f}k")
print(f"   Actual price: ${actual:.2f}k")
print(f"   Difference: ${abs(prediction[0] - actual):.2f}k")

print("\n" + "="*80)
print("✨ REGRESSION TASK COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📦 Next Steps:")
print(f"   1. Use '{model_filename}' in your Streamlit application")
print(f"   2. Load with: model = joblib.load('{model_filename}')")
print(f"   3. Make predictions with: model.predict(new_data)")
print(f"   4. Remember to provide all {X.shape[1]} features in the same order")
print("="*80)

