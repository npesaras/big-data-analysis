"""
Utility functions for model persistence and directory management.
"""

import joblib
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directory(directory_path: Path) -> None:
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory_path: Path to the directory
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {directory_path}")


def save_model(model: Any, filepath: Path, model_name: str = "model") -> bool:
    """
    Save a trained model to disk using joblib.

    Args:
        model: The trained model object
        filepath: Path where the model should be saved
        model_name: Name of the model for logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory(filepath.parent)
        joblib.dump(model, filepath)
        logger.info(f"Model '{model_name}' saved successfully to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model '{model_name}': {str(e)}")
        return False


def load_model(filepath: Path, model_name: str = "model") -> Optional[Any]:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model
        model_name: Name of the model for logging

    Returns:
        The loaded model object, or None if loading fails
    """
    try:
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return None

        model = joblib.load(filepath)
        logger.info(f"Model '{model_name}' loaded successfully from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {str(e)}")
        return None


def save_preprocessors(imputer: Any, scaler: Any, models_dir: Path) -> bool:
    """
    Save preprocessing objects (imputer and scaler).

    Args:
        imputer: The fitted imputer object
        scaler: The fitted scaler object
        models_dir: Directory where preprocessors should be saved

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_directory(models_dir)
        imputer_path = models_dir / "imputer.joblib"
        scaler_path = models_dir / "scaler.joblib"

        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)

        logger.info(f"Preprocessors saved to {models_dir}")
        return True
    except Exception as e:
        logger.error(f"Error saving preprocessors: {str(e)}")
        return False


def load_preprocessors(models_dir: Path) -> tuple:
    """
    Load preprocessing objects (imputer and scaler).

    Args:
        models_dir: Directory where preprocessors are saved

    Returns:
        tuple: (imputer, scaler) or (None, None) if loading fails
    """
    try:
        imputer_path = models_dir / "imputer.joblib"
        scaler_path = models_dir / "scaler.joblib"

        if not imputer_path.exists() or not scaler_path.exists():
            logger.error("Preprocessor files not found")
            return None, None

        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)

        logger.info("Preprocessors loaded successfully")
        return imputer, scaler
    except Exception as e:
        logger.error(f"Error loading preprocessors: {str(e)}")
        return None, None


def get_timestamp() -> str:
    """
    Get current timestamp as a formatted string.

    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger.info(f"Logging configured at {log_level} level")
