"""
Model training functions with cross-validation support.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from typing import Dict, Any, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)


def train_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, float]:
    """
    Train a single model on training data.

    Args:
        model: Untrained model instance
        X_train: Training features
        y_train: Training target

    Returns:
        Tuple of (trained model, training time in seconds)
    """
    start_time = time.time()

    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        model_name = model.__class__.__name__
        logger.info(f"Model '{model_name}' trained in {training_time:.3f} seconds")

        return model, training_time
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, 0.0


def train_with_cross_validation(model: Any,
                                X: pd.DataFrame,
                                y: pd.Series,
                                cv: StratifiedKFold,
                                scoring: List[str] = None) -> Dict:
    """
    Train model with k-fold cross-validation and return metrics.

    Args:
        model: Model instance to train
        X: Feature DataFrame
        y: Target Series
        cv: StratifiedKFold cross-validator
        scoring: List of metrics to compute

    Returns:
        Dictionary with cross-validation results
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    model_name = model.__class__.__name__
    logger.info(f"Starting {cv.get_n_splits()}-fold CV for {model_name}...")

    start_time = time.time()

    try:
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            error_score='raise'
        )

        cv_time = time.time() - start_time

        # Calculate mean and std for each metric
        results = {
            'model_name': model_name,
            'cv_time': cv_time,
            'n_splits': cv.get_n_splits()
        }

        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']

            results[f'{metric}_mean'] = np.mean(test_scores)
            results[f'{metric}_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_scores'] = test_scores.tolist()

        logger.info(f"CV completed for {model_name} in {cv_time:.2f}s - "
                   f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
        return {'model_name': model_name, 'error': str(e)}


def train_all_models(models: Dict[str, Any],
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     cv: StratifiedKFold = None,
                     use_cv: bool = True) -> Dict[str, Dict]:
    """
    Train all models and optionally perform cross-validation.

    Args:
        models: Dictionary of model instances
        X_train: Training features
        y_train: Training target
        cv: StratifiedKFold cross-validator (required if use_cv=True)
        use_cv: Whether to perform cross-validation

    Returns:
        Dictionary with training results for all models
    """
    results = {}
    total_models = len(models)

    logger.info(f"Training {total_models} models...")

    for idx, (model_name, model) in enumerate(models.items(), 1):
        logger.info(f"[{idx}/{total_models}] Training {model_name}...")

        if use_cv and cv is not None:
            # Perform cross-validation
            cv_results = train_with_cross_validation(
                model, X_train, y_train, cv
            )
            results[model_name] = cv_results
        else:
            # Simple training without CV
            trained_model, train_time = train_single_model(model, X_train, y_train)
            results[model_name] = {
                'model': trained_model,
                'training_time': train_time
            }

    logger.info(f"All {total_models} models trained successfully")

    return results


def compare_cv_results(cv_results: Dict[str, Dict],
                      primary_metric: str = 'recall') -> pd.DataFrame:
    """
    Compare cross-validation results across all models.

    Args:
        cv_results: Dictionary with CV results for all models
        primary_metric: Primary metric to sort by

    Returns:
        DataFrame with comparison of all models
    """
    comparison_data = []

    for model_name, results in cv_results.items():
        if 'error' in results:
            continue

        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy_mean', 0),
            'Accuracy_Std': results.get('accuracy_std', 0),
            'Precision': results.get('precision_mean', 0),
            'Recall': results.get('recall_mean', 0),
            'F1-Score': results.get('f1_mean', 0),
            'ROC-AUC': results.get('roc_auc_mean', 0),
            'CV_Time(s)': results.get('cv_time', 0)
        }
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Sort by primary metric
    if primary_metric.capitalize() in df_comparison.columns:
        df_comparison = df_comparison.sort_values(
            primary_metric.capitalize(),
            ascending=False
        )

    logger.info(f"CV results compared - Best {primary_metric}: "
                f"{df_comparison.iloc[0]['Model']} "
                f"({df_comparison.iloc[0][primary_metric.capitalize()]:.4f})")

    return df_comparison


def train_final_model(model: Any,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     model_name: str = "Final Model") -> Tuple[Any, Dict]:
    """
    Train final model on full training data and return training info.

    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training target
        model_name: Name of the model for logging

    Returns:
        Tuple of (trained model, training information dict)
    """
    logger.info(f"Training final {model_name} on full training set...")

    trained_model, train_time = train_single_model(model, X_train, y_train)

    training_info = {
        'model_name': model_name,
        'training_time': train_time,
        'training_samples': len(X_train),
        'n_features': X_train.shape[1],
        'class_distribution': y_train.value_counts().to_dict()
    }

    logger.info(f"Final {model_name} trained on {len(X_train)} samples")

    return trained_model, training_info


def get_training_summary(cv_results: Dict[str, Dict]) -> Dict:
    """
    Generate a summary of training results.

    Args:
        cv_results: Dictionary with CV results for all models

    Returns:
        Dictionary with training summary statistics
    """
    total_models = len(cv_results)
    failed_models = sum(1 for r in cv_results.values() if 'error' in r)
    successful_models = total_models - failed_models

    # Calculate overall statistics
    accuracies = [r['accuracy_mean'] for r in cv_results.values() if 'accuracy_mean' in r]
    recalls = [r['recall_mean'] for r in cv_results.values() if 'recall_mean' in r]
    f1_scores = [r['f1_mean'] for r in cv_results.values() if 'f1_mean' in r]

    summary = {
        'total_models': total_models,
        'successful_models': successful_models,
        'failed_models': failed_models,
        'avg_accuracy': np.mean(accuracies) if accuracies else 0,
        'max_accuracy': np.max(accuracies) if accuracies else 0,
        'min_accuracy': np.min(accuracies) if accuracies else 0,
        'avg_recall': np.mean(recalls) if recalls else 0,
        'max_recall': np.max(recalls) if recalls else 0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0,
        'max_f1': np.max(f1_scores) if f1_scores else 0
    }

    logger.info(f"Training summary: {successful_models}/{total_models} models successful, "
                f"Avg Accuracy: {summary['avg_accuracy']:.4f}")

    return summary


def validate_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> bool:
    """
    Validate training data before model training.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Boolean indicating whether data is valid for training
    """
    issues = []

    # Check for NaN values
    if X_train.isnull().any().any():
        issues.append("Training features contain NaN values")

    if y_train.isnull().any():
        issues.append("Training target contains NaN values")

    # Check for infinite values
    if np.isinf(X_train.values).any():
        issues.append("Training features contain infinite values")

    # Check class balance
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        issues.append("Training target has only one class")

    # Check for sufficient samples
    if len(X_train) < 10:
        issues.append(f"Insufficient training samples: {len(X_train)}")

    if issues:
        for issue in issues:
            logger.error(f"Validation error: {issue}")
        return False

    logger.info("Training data validation passed")
    return True
