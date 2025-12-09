"""
Configuration file for diabetes classification project.
Contains all paths, parameters, and model definitions.
"""

from pathlib import Path
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
DIABETES_DATA = DATA_DIR / "diabetes.csv"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)

# Data configuration
FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET_COLUMN = 'Outcome'

# Columns with zero values that represent missing data
COLS_WITH_ZERO_ISSUES = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

# Model training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
IMPUTATION_STRATEGY = 'median'  # Fixed strategy for consistency

# Model definitions with hyperparameters
MODELS_CONFIG = {
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {
            'random_state': RANDOM_STATE,
            'max_iter': 1000,
            'solver': 'liblinear'
        }
    },
    'Decision Tree': {
        'class': DecisionTreeClassifier,
        'params': {
            'random_state': RANDOM_STATE,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        }
    },
    'Random Forest': {
        'class': RandomForestClassifier,
        'params': {
            'random_state': RANDOM_STATE,
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10
        }
    },
    'Gaussian Naive Bayes': {
        'class': GaussianNB,
        'params': {}
    },
    'K-Nearest Neighbors': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski'
        }
    },
    'Support Vector Machine': {
        'class': SVC,
        'params': {
            'random_state': RANDOM_STATE,
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True
        }
    },
    'AdaBoost': {
        'class': AdaBoostClassifier,
        'params': {
            'random_state': RANDOM_STATE,
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME'
        }
    },
    'Perceptron': {
        'class': Perceptron,
        'params': {
            'random_state': RANDOM_STATE,
            'max_iter': 1000,
            'tol': 1e-3
        }
    },
    'MLP Neural Network': {
        'class': MLPClassifier,
        'params': {
            'random_state': RANDOM_STATE,
            'hidden_layer_sizes': (100, 50),
            'max_iter': 1000,
            'alpha': 0.0001,
            'solver': 'adam'
        }
    }
}

# List of models to test (all 9)
MODELS_TO_TEST = list(MODELS_CONFIG.keys())

# Evaluation metrics priority (for medical diagnosis, prioritize Recall)
PRIMARY_METRIC = 'recall'  # Minimize false negatives
SECONDARY_METRICS = ['f1', 'accuracy', 'precision', 'roc_auc']

# Streamlit configuration
APP_TITLE = "🏥 Diabetes Classification System"
APP_ICON = "🏥"
PAGE_CONFIG = {
    'page_title': APP_TITLE,
    'page_icon': APP_ICON,
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Input validation ranges (based on dataset statistics)
INPUT_RANGES = {
    'Pregnancies': {'min': 0, 'max': 17, 'normal_range': (0, 10)},
    'Glucose': {'min': 44, 'max': 199, 'normal_range': (70, 140)},
    'BloodPressure': {'min': 40, 'max': 122, 'normal_range': (60, 90)},
    'SkinThickness': {'min': 7, 'max': 99, 'normal_range': (10, 50)},
    'Insulin': {'min': 14, 'max': 846, 'normal_range': (16, 166)},
    'BMI': {'min': 15.0, 'max': 67.1, 'normal_range': (18.5, 30.0)},
    'DiabetesPedigreeFunction': {'min': 0.078, 'max': 2.420, 'normal_range': (0.2, 1.0)},
    'Age': {'min': 21, 'max': 81, 'normal_range': (21, 65)}
}
