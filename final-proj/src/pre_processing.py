"""
Data preprocessing pipeline: handle zeros, imputation, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def handle_zero_values(df: pd.DataFrame, cols_to_fix: List[str]) -> pd.DataFrame:
    """
    Replace zero values with NaN in specified columns.
    These zeros represent biologically impossible values (missing data).

    Args:
        df: Input DataFrame
        cols_to_fix: List of column names where zeros should be replaced with NaN

    Returns:
        DataFrame with zeros replaced by NaN
    """
    df_clean = df.copy()

    for col in cols_to_fix:
        if col in df_clean.columns:
            zero_count = (df_clean[col] == 0).sum()
            if zero_count > 0:
                df_clean[col] = df_clean[col].replace(0, np.nan)
                logger.info(f"Replaced {zero_count} zeros with NaN in column '{col}'")

    return df_clean


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values using the specified strategy.

    Args:
        df: Input DataFrame with missing values
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')

    Returns:
        Tuple of (imputed DataFrame, fitted imputer object)
    """
    imputer = SimpleImputer(strategy=strategy)

    # Fit and transform the data
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    # Count how many values were imputed
    missing_before = df.isnull().sum().sum()
    logger.info(f"Imputed {missing_before} missing values using '{strategy}' strategy")

    return df_imputed, imputer


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features using StandardScaler (mean=0, std=1).

    Args:
        X: Input feature DataFrame

    Returns:
        Tuple of (scaled DataFrame, fitted scaler object)
    """
    scaler = StandardScaler()

    # Fit and transform the data
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    logger.info("Features scaled using StandardScaler (mean=0, std=1)")

    return X_scaled, scaler


def preprocess_pipeline(df: pd.DataFrame,
                       feature_columns: List[str],
                       target_column: str,
                       cols_with_zeros: List[str],
                       imputation_strategy: str = 'median') -> Tuple[pd.DataFrame, pd.Series, SimpleImputer, StandardScaler]:
    """
    Complete preprocessing pipeline: handle zeros → impute → scale.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of target variable
        cols_with_zeros: Columns where zeros should be treated as missing
        imputation_strategy: Strategy for imputation

    Returns:
        Tuple of (X_scaled, y, imputer, scaler)
    """
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Separate features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Step 2: Handle zero values (replace with NaN)
    X_with_nan = handle_zero_values(X, cols_with_zeros)

    # Step 3: Impute missing values
    X_imputed, imputer = impute_missing_values(X_with_nan, strategy=imputation_strategy)

    # Step 4: Scale features
    X_scaled, scaler = scale_features(X_imputed)

    logger.info("Preprocessing pipeline completed successfully")

    return X_scaled, y, imputer, scaler


def apply_preprocessing(X: pd.DataFrame,
                       cols_with_zeros: List[str],
                       imputer: SimpleImputer,
                       scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply fitted preprocessing transformations to new data.
    Use this for test data or new predictions.

    Args:
        X: Input feature DataFrame
        cols_with_zeros: Columns where zeros should be treated as missing
        imputer: Fitted imputer object
        scaler: Fitted scaler object

    Returns:
        Preprocessed DataFrame
    """
    # Step 1: Handle zeros
    X_with_nan = handle_zero_values(X, cols_with_zeros)

    # Step 2: Apply imputation
    X_imputed = pd.DataFrame(
        imputer.transform(X_with_nan),
        columns=X.columns,
        index=X.index
    )

    # Step 3: Apply scaling
    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index
    )

    logger.info("Preprocessing applied to new data")

    return X_scaled


def get_preprocessing_summary(df_original: pd.DataFrame,
                             df_processed: pd.DataFrame,
                             cols_with_zeros: List[str]) -> dict:
    """
    Generate a summary of preprocessing transformations.

    Args:
        df_original: Original DataFrame before preprocessing
        df_processed: DataFrame after preprocessing
        cols_with_zeros: Columns that had zeros replaced

    Returns:
        Dictionary with preprocessing summary
    """
    summary = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'zeros_handled': {},
        'missing_before': df_original.isnull().sum().to_dict(),
        'missing_after': 0,  # Should be 0 after imputation
        'scaling_applied': True
    }

    # Count zeros that were handled
    for col in cols_with_zeros:
        if col in df_original.columns:
            zero_count = (df_original[col] == 0).sum()
            summary['zeros_handled'][col] = int(zero_count)

    return summary


def inverse_transform_features(X_scaled: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Inverse transform scaled features back to original scale.
    Useful for interpretation and visualization.

    Args:
        X_scaled: Scaled feature DataFrame
        scaler: Fitted scaler object

    Returns:
        DataFrame with features in original scale
    """
    X_original = pd.DataFrame(
        scaler.inverse_transform(X_scaled),
        columns=X_scaled.columns,
        index=X_scaled.index
    )

    logger.info("Features inverse transformed to original scale")

    return X_original
