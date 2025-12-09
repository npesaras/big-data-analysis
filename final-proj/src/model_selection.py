"""
Model selection: Create and configure all 9 classification algorithms.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def create_all_models(models_config: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create instances of all classification models with configured hyperparameters.

    Args:
        models_config: Dictionary containing model configurations from config.py

    Returns:
        Dictionary mapping model names to instantiated model objects
    """
    models = {}

    for model_name, config in models_config.items():
        try:
            model_class = config['class']
            params = config['params']
            models[model_name] = model_class(**params)
            logger.info(f"Created model: {model_name}")
        except Exception as e:
            logger.error(f"Error creating model '{model_name}': {str(e)}")

    logger.info(f"Successfully created {len(models)} models")
    return models


def create_single_model(model_name: str, models_config: Dict[str, Dict]) -> Any:
    """
    Create a single model instance by name.

    Args:
        model_name: Name of the model to create
        models_config: Dictionary containing model configurations

    Returns:
        Instantiated model object, or None if model not found
    """
    if model_name not in models_config:
        logger.error(f"Model '{model_name}' not found in configuration")
        return None

    try:
        config = models_config[model_name]
        model_class = config['class']
        params = config['params']
        model = model_class(**params)
        logger.info(f"Created model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error creating model '{model_name}': {str(e)}")
        return None


def get_model_descriptions() -> Dict[str, str]:
    """
    Get human-readable descriptions of all models.

    Returns:
        Dictionary mapping model names to their descriptions
    """
    descriptions = {
        'Logistic Regression':
            'Linear model using logistic function for binary classification. '
            'Fast, interpretable, and works well as a baseline.',

        'Decision Tree':
            'Tree-based model that splits data based on feature values. '
            'Interpretable but can overfit without proper constraints.',

        'Random Forest':
            'Ensemble of decision trees using bagging. '
            'Reduces overfitting and generally provides better performance than single trees.',

        'Gaussian Naive Bayes':
            'Probabilistic classifier based on Bayes theorem with Gaussian distribution assumption. '
            'Fast and works well with small datasets.',

        'K-Nearest Neighbors':
            'Instance-based learning that classifies based on k nearest neighbors. '
            'Non-parametric and can capture complex patterns.',

        'Support Vector Machine':
            'Finds optimal hyperplane to separate classes. '
            'Effective in high-dimensional spaces with RBF kernel.',

        'AdaBoost':
            'Ensemble method that combines weak learners iteratively. '
            'Focuses on misclassified samples to improve performance.',

        'Perceptron':
            'Simple linear classifier inspired by biological neurons. '
            'Fast but only works for linearly separable data.',

        'MLP Neural Network':
            'Multi-layer perceptron with hidden layers. '
            'Can learn complex non-linear patterns through backpropagation.'
    }

    return descriptions


def get_model_hyperparameters(models_config: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Extract hyperparameters for all models.

    Args:
        models_config: Dictionary containing model configurations

    Returns:
        Dictionary mapping model names to their hyperparameters
    """
    hyperparameters = {}

    for model_name, config in models_config.items():
        hyperparameters[model_name] = config['params'].copy()

    return hyperparameters


def get_model_complexity_info() -> Dict[str, Dict]:
    """
    Get information about model complexity and computational requirements.

    Returns:
        Dictionary with complexity information for each model
    """
    complexity = {
        'Logistic Regression': {
            'training_speed': 'Fast',
            'prediction_speed': 'Very Fast',
            'memory_usage': 'Low',
            'interpretability': 'High',
            'complexity': 'Low'
        },
        'Decision Tree': {
            'training_speed': 'Fast',
            'prediction_speed': 'Fast',
            'memory_usage': 'Low',
            'interpretability': 'High',
            'complexity': 'Medium'
        },
        'Random Forest': {
            'training_speed': 'Medium',
            'prediction_speed': 'Medium',
            'memory_usage': 'Medium',
            'interpretability': 'Medium',
            'complexity': 'Medium-High'
        },
        'Gaussian Naive Bayes': {
            'training_speed': 'Very Fast',
            'prediction_speed': 'Very Fast',
            'memory_usage': 'Low',
            'interpretability': 'Medium',
            'complexity': 'Low'
        },
        'K-Nearest Neighbors': {
            'training_speed': 'Very Fast',
            'prediction_speed': 'Slow',
            'memory_usage': 'High',
            'interpretability': 'Low',
            'complexity': 'Low'
        },
        'Support Vector Machine': {
            'training_speed': 'Slow',
            'prediction_speed': 'Fast',
            'memory_usage': 'Medium',
            'interpretability': 'Low',
            'complexity': 'High'
        },
        'AdaBoost': {
            'training_speed': 'Medium',
            'prediction_speed': 'Medium',
            'memory_usage': 'Medium',
            'interpretability': 'Medium',
            'complexity': 'Medium'
        },
        'Perceptron': {
            'training_speed': 'Very Fast',
            'prediction_speed': 'Very Fast',
            'memory_usage': 'Low',
            'interpretability': 'High',
            'complexity': 'Low'
        },
        'MLP Neural Network': {
            'training_speed': 'Slow',
            'prediction_speed': 'Fast',
            'memory_usage': 'Medium-High',
            'interpretability': 'Low',
            'complexity': 'High'
        }
    }

    return complexity


def get_recommended_use_cases() -> Dict[str, List[str]]:
    """
    Get recommended use cases for each model.

    Returns:
        Dictionary mapping model names to list of recommended scenarios
    """
    use_cases = {
        'Logistic Regression': [
            'Baseline model',
            'When interpretability is crucial',
            'Linear relationships in data',
            'Quick prototyping'
        ],
        'Decision Tree': [
            'When feature interactions are important',
            'Need for interpretability',
            'Handling non-linear relationships'
        ],
        'Random Forest': [
            'General-purpose classification',
            'Handling complex patterns',
            'Reducing overfitting',
            'Feature importance analysis'
        ],
        'Gaussian Naive Bayes': [
            'Small datasets',
            'Real-time predictions needed',
            'When features are relatively independent'
        ],
        'K-Nearest Neighbors': [
            'Non-linear decision boundaries',
            'Small to medium datasets',
            'When similar instances should have similar predictions'
        ],
        'Support Vector Machine': [
            'High-dimensional data',
            'Clear margin of separation',
            'When accuracy is more important than speed'
        ],
        'AdaBoost': [
            'Improving weak classifiers',
            'Handling imbalanced data',
            'When high accuracy is needed'
        ],
        'Perceptron': [
            'Linearly separable data',
            'Online learning scenarios',
            'Simple baseline model'
        ],
        'MLP Neural Network': [
            'Complex non-linear patterns',
            'Large datasets',
            'When maximum accuracy is priority'
        ]
    }

    return use_cases


def validate_models_config(models_config: Dict[str, Dict]) -> bool:
    """
    Validate that models configuration is properly structured.

    Args:
        models_config: Dictionary containing model configurations

    Returns:
        Boolean indicating whether configuration is valid
    """
    required_keys = ['class', 'params']

    for model_name, config in models_config.items():
        for key in required_keys:
            if key not in config:
                logger.error(f"Model '{model_name}' missing required key: '{key}'")
                return False

        if not callable(config['class']):
            logger.error(f"Model '{model_name}' class is not callable")
            return False

        if not isinstance(config['params'], dict):
            logger.error(f"Model '{model_name}' params must be a dictionary")
            return False

    logger.info(f"Models configuration validated successfully ({len(models_config)} models)")
    return True
