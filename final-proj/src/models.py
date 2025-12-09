"""
Model training and evaluation functions for Lab 2.

This module provides reusable functions for creating, training,
and evaluating machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_classification_pipeline(random_state: int = 42) -> Pipeline:
    """
    Create a classification pipeline for diabetes prediction.

    Args:
        random_state: Random state for reproducibility

    Returns:
        sklearn Pipeline for classification
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    logger.info("Created classification pipeline: Scaler → LogisticRegression")
    return pipeline

def create_regression_pipeline() -> Pipeline:
    """
    Create a regression pipeline for housing price prediction.

    Returns:
        sklearn Pipeline for regression
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    logger.info("Created regression pipeline: Imputer → Scaler → LinearRegression")
    return pipeline

def evaluate_classification_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluate classification model using multiple cross-validation strategies.

    Args:
        pipeline: sklearn Pipeline to evaluate
        X: Features DataFrame
        y: Target Series

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating classification model...")

    results = {}

    # Model A: K-Fold Cross-Validation (10 splits)
    logger.info("Running K-Fold Cross-Validation (10 splits)...")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    y_pred_kfold = cross_val_predict(pipeline, X, y, cv=kfold, method='predict')
    y_pred_proba_kfold = cross_val_predict(pipeline, X, y, cv=kfold, method='predict_proba')

    results['kfold_accuracy'] = accuracy_score(y, y_pred_kfold)
    results['kfold_logloss'] = log_loss(y, y_pred_proba_kfold)
    results['kfold_roc_auc'] = roc_auc_score(y, y_pred_proba_kfold[:, 1])

    # Model B: Leave-One-Out Cross-Validation
    logger.info("Running Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()

    y_pred_loo = cross_val_predict(pipeline, X, y, cv=loo, method='predict')
    y_pred_proba_loo = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba')

    results['loo_accuracy'] = accuracy_score(y, y_pred_loo)
    results['loo_logloss'] = log_loss(y, y_pred_proba_loo)
    results['loo_roc_auc'] = roc_auc_score(y, y_pred_proba_loo[:, 1])

    logger.info("Classification evaluation completed")
    return results

def evaluate_regression_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluate regression model using repeated cross-validation.

    Args:
        pipeline: sklearn Pipeline to evaluate
        X: Features DataFrame
        y: Target Series

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating regression model...")

    # Use RepeatedKFold for robust evaluation
    rkfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    logger.info("Running Repeated K-Fold Cross-Validation (10 splits × 3 repeats)...")

    # Get cross-validation scores
    mse_scores = cross_val_score(pipeline, X, y, cv=rkfold, scoring='neg_mean_squared_error', n_jobs=-1)
    mae_scores = cross_val_score(pipeline, X, y, cv=rkfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores = cross_val_score(pipeline, X, y, cv=rkfold, scoring='r2', n_jobs=-1)

    # Convert negative scores back to positive
    mse_scores = -mse_scores
    mae_scores = -mae_scores

    results = {
        'mse_mean': mse_scores.mean(),
        'mse_std': mse_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'rmse_mean': np.sqrt(mse_scores).mean(),
        'rmse_std': np.sqrt(mse_scores).std()
    }

    logger.info("Regression evaluation completed")
    return results

def print_classification_results(metrics: Dict[str, Any]) -> None:
    """Print formatted classification evaluation results."""
    print("\n" + "="*80)
    print("CLASSIFICATION MODEL EVALUATION RESULTS")
    print("="*80)

    print(f"\n📊 K-FOLD CROSS-VALIDATION (10 splits):")
    print(f"   Accuracy: {metrics['kfold_accuracy']:.4f}")
    print(f"   Log Loss: {metrics['kfold_logloss']:.4f}")
    print(f"   ROC AUC: {metrics['kfold_roc_auc']:.4f}")

    print(f"\n🔄 LEAVE-ONE-OUT CROSS-VALIDATION:")
    print(f"   Accuracy: {metrics['loo_accuracy']:.4f}")
    print(f"   Log Loss: {metrics['loo_logloss']:.4f}")
    print(f"   ROC AUC: {metrics['loo_roc_auc']:.4f}")

    # Determine best model
    kfold_accuracy = metrics['kfold_accuracy']
    loo_accuracy = metrics['loo_accuracy']

    if kfold_accuracy >= loo_accuracy:
        print(f"\n🏆 SELECTED MODEL: K-FOLD CROSS-VALIDATION")
        print("   Justification: Better computational efficiency with comparable performance")
    else:
        print(f"\n🏆 SELECTED MODEL: LEAVE-ONE-OUT CROSS-VALIDATION")
        print("   Justification: Slightly better performance despite higher computational cost")

def print_regression_results(metrics: Dict[str, Any]) -> None:
    """Print formatted regression evaluation results."""
    print("\n" + "="*80)
    print("REGRESSION MODEL EVALUATION RESULTS")
    print("="*80)

    print(f"\n📊 REPEATED K-FOLD CROSS-VALIDATION (10×3):")
    print(f"   MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
    print(f"   MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
    print(f"   RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"   R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")

    print(f"\n💡 INTERPRETATION:")
    mae = metrics['mae_mean']
    r2 = metrics['r2_mean']
    print(f"   Average prediction error: ${mae*1000:.2f}")
    print(f"   Model explains {r2:.2f} of price variation")

    print(f"\n🏆 SELECTED MODEL: REPEATED K-FOLD CROSS-VALIDATION")
    print("   Justification: Robust evaluation with confidence intervals")