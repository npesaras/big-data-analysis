#!/usr/bin/env python3
"""
Refactored Regression Training Script for Housing Price Prediction

This script demonstrates the separated architecture:
- Data cleaning logic is in src/data_cleaning.py
- Model training logic is in src/models.py
- Configuration is in src/config.py
- Utilities are in src/utils.py

Usage:
    python scripts/train_regression.py
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_cleaning import load_housing_data, validate_data
from src.models import create_regression_pipeline, evaluate_regression_model, print_regression_results
from src.utils import setup_logging, save_model, print_header, validate_file_exists

def main():
    """Main training function for housing price regression."""
    # Setup logging
    setup_logging()

    print_header("HOUSING PRICE REGRESSION TRAINING")

    try:
        # Validate data file exists
        validate_file_exists(str(Config.HOUSING_DATA), "Housing dataset")

        # Load and preprocess data
        print("\n📊 Loading housing data...")
        X, y = load_housing_data(str(Config.HOUSING_DATA))

        # Validate data
        validate_data(X, y, "housing regression")

        # Create and evaluate model
        print("\n🔧 Creating regression pipeline...")
        pipeline = create_regression_pipeline()

        print("\n📈 Evaluating model...")
        metrics = evaluate_regression_model(pipeline, X, y)

        # Print results
        print_regression_results(metrics)

        # Train final model on entire dataset
        print("\n🏃 Training final model on entire dataset...")
        pipeline.fit(X, y)

        # Save the model
        save_model(pipeline, str(Config.REGRESSION_MODEL))

        print_header("REGRESSION TRAINING COMPLETED SUCCESSFULLY")
        print(f"\n📦 Model saved to: {Config.REGRESSION_MODEL}")
        print(f"📊 Final R² score: {metrics['r2_mean']:.4f}")
        print(f"📏 Final MAE: ${metrics['mae_mean']:.2f}k")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()