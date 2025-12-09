"""
Utility functions for Lab 2 ML pipeline.

This module provides helper functions for logging, file operations,
and other common utilities.
"""

import logging
import joblib
from pathlib import Path
from typing import Any
import sys

def setup_logging(level: str = 'INFO', log_file: str = None) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        ensure_directory(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def ensure_directory(filepath: str) -> None:
    """
    Ensure the directory for a file path exists.

    Args:
        filepath: File path (directory will be created if needed)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

def save_model(model: Any, filepath: str) -> None:
    """
    Save a model to disk with proper error handling.

    Args:
        model: Model object to save
        filepath: Path where to save the model
    """
    try:
        ensure_directory(filepath)
        joblib.dump(model, filepath)
        print(f"✅ Model saved successfully to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model to {filepath}: {e}")
        raise

def load_model(filepath: str) -> Any:
    """
    Load a model from disk with proper error handling.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        print(f"✅ Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {filepath}: {e}")
        raise

def validate_file_exists(filepath: str, description: str = "file") -> None:
    """
    Validate that a file exists.

    Args:
        filepath: Path to check
        description: Description of the file for error messages

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"{description} not found: {filepath}")

def print_header(title: str, width: int = 80) -> None:
    """
    Print a formatted header.

    Args:
        title: Header text
        width: Width of the header
    """
    print("\n" + "="*width)
    print(f"{title:^{width}}")
    print("="*width)

def print_section(title: str) -> None:
    """
    Print a section header.

    Args:
        title: Section title
    """
    print(f"\n{title}")
    print("-" * len(title))

def format_number(num: float, decimals: int = 4) -> str:
    """
    Format a number for display.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{num:.{decimals}f}"