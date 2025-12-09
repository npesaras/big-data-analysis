"""
Configuration settings for Lab 2 ML pipeline.
Centralized configuration to avoid hardcoded paths and parameters.
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    """Configuration class for all paths and parameters."""

    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    DIABETES_DATA = DATA_DIR / "diabetes.csv"
    HOUSING_DATA = DATA_DIR / "house-data.csv"

    # Model paths
    MODELS_DIR = PROJECT_ROOT / "models"
    CLASSIFICATION_MODEL = MODELS_DIR / "classification_model.joblib"
    REGRESSION_MODEL = MODELS_DIR / "regression_model.joblib"

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Cross-validation parameters
    CV_FOLDS = 10
    CV_REPEATS = 3

    # Model hyperparameters
    LOGISTIC_REGRESSION_PARAMS = {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }

    # Logging
    LOG_LEVEL = 'INFO'