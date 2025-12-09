#!/usr/bin/env python3
"""
Refactored Classification Training Script for Diabetes Prediction

This script demonstrates the separated architecture:
- Data cleaning logic is in src/data_cleaning.py
- Model training logic is in src/models.py
- Configuration is in src/config.py
- Utilities are in src/utils.py

Usage:
    python scripts/train_classification.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data_cleaning import load_diabetes_data, validate_data
from src.models import create_classification_pipeline, evaluate_classification_model, print_classification_results
from src.utils import setup_logging, save_model, print_header, validate_file_exists

def main():
    """Main training function for diabetes classification."""
    # Setup logging
    setup_logging()

    print_header("DIABETES CLASSIFICATION TRAINING")

    try:
        # Validate data file exists
        validate_file_exists(str(Config.DIABETES_DATA), "Diabetes dataset")

        # Load and preprocess data
        print("\n📊 Loading diabetes data...")
        X, y = load_diabetes_data(str(Config.DIABETES_DATA))

        # Validate data
        validate_data(X, y, "diabetes classification")

        # Create and evaluate model
        print("\n🔧 Creating classification pipeline...")
        pipeline = create_classification_pipeline(Config.RANDOM_STATE)

        print("\n📈 Evaluating model...")
        metrics = evaluate_classification_model(pipeline, X, y)

        # Print results
        print_classification_results(metrics)

        # Train final model on entire dataset
        print("\n🏃 Training final model on entire dataset...")
        pipeline.fit(X, y)

        # Save the model
        save_model(pipeline, str(Config.CLASSIFICATION_MODEL))

        print_header("CLASSIFICATION TRAINING COMPLETED SUCCESSFULLY")
        print(f"\n📦 Model saved to: {Config.CLASSIFICATION_MODEL}")
        print(f"📊 Final accuracy: {metrics['kfold_accuracy']:.4f}")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()