"""
Data loading and preprocessing functions for Lab 2.

This module separates data cleaning logic from model training,
providing reusable functions for loading and preprocessing datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def load_diabetes_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and perform initial preprocessing of diabetes dataset.

    Args:
        filepath: Path to the diabetes.csv file

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    try:
        logger.info(f"Loading diabetes data from {filepath}")
        df = pd.read_csv(filepath)

        # Basic validation
        if 'Outcome' not in df.columns:
            raise ValueError("Diabetes dataset must contain 'Outcome' column")

        # Split features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        logger.info(f"Diabetes data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    except FileNotFoundError:
        logger.error(f"Diabetes data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading diabetes data: {e}")
        raise

def load_housing_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and perform initial preprocessing of housing dataset.

    Args:
        filepath: Path to the house-data.csv file

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    try:
        logger.info(f"Loading housing data from {filepath}")
        df = pd.read_csv(filepath)

        # Basic validation
        if 'MEDV' not in df.columns:
            raise ValueError("Housing dataset must contain 'MEDV' column")

        # Split features and target
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']

        # Check for missing values
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")

        logger.info(f"Housing data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target range: ${y.min():.1f}k - ${y.max():.1f}k (median: ${y.mean():.1f}k)")

        return X, y

    except FileNotFoundError:
        logger.error(f"Housing data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading housing data: {e}")
        raise

def create_preprocessing_pipeline() -> Pipeline:
    """
    Create a reusable preprocessing pipeline.

    Returns:
        sklearn Pipeline with imputation and scaling
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    logger.info("Created preprocessing pipeline: Imputer → Scaler")
    return pipeline

def validate_data(X: pd.DataFrame, y: pd.Series, task: str = "unknown") -> None:
    """
    Perform basic data validation.

    Args:
        X: Features DataFrame
        y: Target Series
        task: Task name for logging

    Raises:
        ValueError: If data validation fails
    """
    if X.empty:
        raise ValueError(f"Features data is empty for {task} task")

    if y.empty:
        raise ValueError(f"Target data is empty for {task} task")

    if len(X) != len(y):
        raise ValueError(f"Features ({len(X)}) and target ({len(y)}) have different lengths")

    # Check for NaN values in target
    if y.isnull().any():
        raise ValueError(f"Target contains NaN values for {task} task")

    logger.info(f"Data validation passed for {task} task")