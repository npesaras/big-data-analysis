"""
Data splitting functions with stratification for train-test and cross-validation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with optional stratification.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class balance in splits

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    logger.info(f"Data split: Train={len(X_train)} ({(1-test_size)*100:.0f}%), "
                f"Test={len(X_test)} ({test_size*100:.0f}%)")

    if stratify:
        train_dist = y_train.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()
        logger.info(f"Train class distribution: {train_dist}")
        logger.info(f"Test class distribution: {test_dist}")

    return X_train, X_test, y_train, y_test


def create_cross_validation_folds(n_splits: int = 10,
                                  random_state: int = 42,
                                  shuffle: bool = True) -> StratifiedKFold:
    """
    Create a StratifiedKFold cross-validator for maintaining class balance.

    Args:
        n_splits: Number of folds (default 10 for 10-fold CV)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting

    Returns:
        StratifiedKFold object for cross-validation
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    logger.info(f"Created {n_splits}-fold stratified cross-validator")

    return cv


def get_split_info(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series) -> Dict:
    """
    Get detailed information about the train-test split.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target

    Returns:
        Dictionary with split information
    """
    total_samples = len(X_train) + len(X_test)

    info = {
        'total_samples': total_samples,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_percentage': round(len(X_train) / total_samples * 100, 2),
        'test_percentage': round(len(X_test) / total_samples * 100, 2),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'train_class_distribution': y_train.value_counts().to_dict(),
        'test_class_distribution': y_test.value_counts().to_dict(),
        'train_class_ratio': round(
            y_train.value_counts()[0] / y_train.value_counts()[1], 2
        ),
        'test_class_ratio': round(
            y_test.value_counts()[0] / y_test.value_counts()[1], 2
        )
    }

    logger.info(f"Split info generated - Total: {total_samples}, "
                f"Train: {info['train_samples']}, Test: {info['test_samples']}")

    return info


def validate_split(y_train: pd.Series, y_test: pd.Series,
                   tolerance: float = 0.05) -> bool:
    """
    Validate that the train-test split maintains similar class distributions.

    Args:
        y_train: Training target
        y_test: Testing target
        tolerance: Maximum allowed difference in class proportions (default 5%)

    Returns:
        Boolean indicating whether split is valid
    """
    train_props = y_train.value_counts(normalize=True).sort_index()
    test_props = y_test.value_counts(normalize=True).sort_index()

    max_diff = abs(train_props - test_props).max()

    is_valid = max_diff <= tolerance

    if is_valid:
        logger.info(f"Split validation passed (max difference: {max_diff:.4f})")
    else:
        logger.warning(f"Split validation failed (max difference: {max_diff:.4f} > {tolerance})")

    return is_valid


def get_cv_split_info(cv: StratifiedKFold, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Get information about cross-validation splits.

    Args:
        cv: StratifiedKFold cross-validator
        X: Feature DataFrame
        y: Target Series

    Returns:
        Dictionary with CV split information
    """
    fold_info = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        fold_info.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_class_dist': y_train_fold.value_counts().to_dict(),
            'val_class_dist': y_val_fold.value_counts().to_dict()
        })

    cv_info = {
        'n_splits': cv.get_n_splits(),
        'total_samples': len(X),
        'folds': fold_info
    }

    logger.info(f"CV split info generated for {cv.get_n_splits()} folds")

    return cv_info


def create_temporal_split(X: pd.DataFrame,
                         y: pd.Series,
                         train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a temporal/sequential split without shuffling.
    Useful when data has temporal ordering.

    Args:
        X: Feature DataFrame
        y: Target Series
        train_ratio: Proportion of data for training

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Temporal split created: Train={len(X_train)}, Test={len(X_test)}")

    return X_train, X_test, y_train, y_test
