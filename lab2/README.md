# ITD105 - Big Data Analytics Lab Exercise #2

A comprehensive implementation of classification and regression models using advanced resampling techniques, featuring a modular architecture that separates data processing from model training for better maintainability and scalability.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Datasets](#models--datasets)
- [API Reference](#api-reference)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🔬 Machine Learning Tasks

- **Diabetes Classification**: Binary classification using PIMA Indians Diabetes Dataset
- **Housing Price Regression**: Regression analysis using Boston Housing Dataset

### 🏗️ Advanced Architecture

- **Modular Design**: Complete separation of data cleaning and model training
- **Configuration Management**: Centralized path and parameter management
- **Error Handling**: Comprehensive logging and exception management
- **Type Safety**: Full type hints throughout the codebase

### 📊 Resampling Techniques

- **K-Fold Cross-Validation** (10 folds) for robust classification evaluation
- **Leave-One-Out Cross-Validation** for maximum training data utilization
- **Repeated K-Fold Cross-Validation** (10×3) for stable regression metrics

### 🎯 Performance Metrics

- **Classification**: Accuracy, Log Loss, ROC AUC, Confusion Matrix
- **Regression**: MAE, MSE, RMSE, R² with confidence intervals

### 🌐 Web Interface

- **Interactive Streamlit App**: User-friendly prediction interface
- **Real-time Predictions**: Instant model inference
- **Responsive Design**: Mobile and desktop optimized

---

## 🏗️ Architecture

### Modular Design Philosophy

```
lab2/
├── src/                    # Core business logic
│   ├── __init__.py        # Package metadata
│   ├── config.py          # Centralized configuration
│   ├── data_cleaning.py   # Data loading & preprocessing
│   ├── models.py          # Model creation & evaluation
│   └── utils.py           # Helper functions
├── scripts/               # Training orchestration
│   ├── train_classification.py
│   └── train_regression.py
├── models/                # Trained model artifacts
├── data/                  # Raw datasets
└── main.py               # Streamlit application
```

### Separation of Concerns

| Component | Responsibility | Key Functions |
|-----------|----------------|----------------|
| `config.py` | Configuration management | Path definitions, hyperparameters |
| `data_cleaning.py` | Data processing | `load_diabetes_data()`, `load_housing_data()` |
| `models.py` | ML operations | `create_*_pipeline()`, `evaluate_*_model()` |
| `utils.py` | Utilities | `load_model()`, `save_model()`, logging |
| `scripts/` | Training orchestration | End-to-end training workflows |

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.13 or higher
- **Package Manager**: uv (recommended) or pip

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd lab2
   ```

2. **Install dependencies**

   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Train models** (optional - pre-trained models included)

   ```bash
   # Train classification model
   uv run python scripts/train_classification.py

   # Train regression model
   uv run python scripts/train_regression.py
   ```

4. **Launch the application**

   ```bash
   uv run streamlit run main.py
   ```

---

## 📖 Usage

### Web Application

The Streamlit application provides two main prediction interfaces:

#### 🏥 Diabetes Prediction

- Input patient diagnostic measurements
- Get real-time diabetes risk assessment
- View confidence scores and clinical interpretation

#### 🏠 Housing Price Prediction

- Input property characteristics
- Receive price estimates with confidence intervals
- Analyze key factors influencing property values

### Programmatic Usage

```python
from src.config import Config
from src.data_cleaning import load_diabetes_data
from src.models import create_classification_pipeline
from src.utils import load_model

# Load configuration
print(f"Data directory: {Config.DATA_DIR}")

# Load and preprocess data
X, y = load_diabetes_data(str(Config.DIABETES_DATA))

# Create and train model
pipeline = create_classification_pipeline()
pipeline.fit(X, y)

# Make predictions
predictions = pipeline.predict(X)
```

### Training Scripts

```bash
# Train individual models
python scripts/train_classification.py
python scripts/train_regression.py

# Models are automatically saved to models/ directory
```

---

## 📁 Project Structure

```
lab2/
├── .python-version          # Python version specification
├── pyproject.toml          # Project configuration & dependencies
├── README.md               # This file
├── main.py                 # Streamlit web application
├── regression.py           # Legacy regression implementation (deprecated)
│
├── src/                    # Core package
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── data_cleaning.py   # Data loading & preprocessing
│   ├── models.py          # ML model creation & evaluation
│   └── utils.py           # Utility functions
│
├── scripts/               # Training scripts
│   ├── train_classification.py
│   └── train_regression.py
│
├── models/                # Trained model artifacts
│   ├── classification_model.joblib
│   └── regression_model.joblib
│
├── data/                  # Datasets
│   ├── diabetes.csv      # PIMA Indians Diabetes Dataset
│   └── house-data.csv    # Boston Housing Dataset
│
├── data-training/         # Legacy training files (deprecated)
│   ├── classification.py
│   └── regression.py
│
└── docs/                  # Documentation
```

---

## 🤖 Models & Datasets

### Classification Model

- **Algorithm**: Logistic Regression with Standard Scaling
- **Dataset**: PIMA Indians Diabetes Dataset (768 samples, 8 features)
- **Evaluation**: K-Fold vs Leave-One-Out Cross-Validation
- **Performance**: ~77% accuracy with ROC AUC > 0.8

### Regression Model

- **Algorithm**: Linear Regression with Imputation and Scaling
- **Dataset**: Boston Housing Dataset (506 samples, 13 features)
- **Evaluation**: Repeated K-Fold Cross-Validation (10×3)
- **Performance**: R² ~0.70, MAE ~$3,400

### Dataset Sources

- **Diabetes Data**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/34/diabetes)
- **Housing Data**: [Scikit-learn Boston Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)

---

## 📚 API Reference

### Configuration (`src.config`)

```python
from src.config import Config

# Access paths
Config.DATA_DIR           # Data directory
Config.DIABETES_DATA      # Diabetes dataset path
Config.CLASSIFICATION_MODEL  # Classification model path

# Access parameters
Config.RANDOM_STATE       # Random seed
Config.CV_FOLDS          # Cross-validation folds
```

### Data Cleaning (`src.data_cleaning`)

```python
from src.data_cleaning import load_diabetes_data, load_housing_data

# Load datasets
X_diabetes, y_diabetes = load_diabetes_data(filepath)
X_housing, y_housing = load_housing_data(filepath)
```

### Models (`src.models`)

```python
from src.models import create_classification_pipeline, evaluate_classification_model

# Create pipelines
clf_pipeline = create_classification_pipeline()
reg_pipeline = create_regression_pipeline()

# Evaluate models
metrics = evaluate_classification_model(pipeline, X, y)
results = evaluate_regression_model(pipeline, X, y)
```

### Utilities (`src.utils`)

```python
from src.utils import load_model, save_model, setup_logging

# Model persistence
model = load_model(filepath)
save_model(model, filepath)

# Logging setup
setup_logging(level='INFO', log_file='training.log')
```

---

## 🛠️ Development

### Environment Setup

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Code Quality

```bash
# Run tests (if implemented)
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/

# Formatting
uv run black src/
uv run isort src/
```

### Adding New Features

1. **Data Processing**: Add functions to `src/data_cleaning.py`
2. **Model Logic**: Add functions to `src/models.py`
3. **Configuration**: Update `src/config.py`
4. **Utilities**: Add helpers to `src/utils.py`

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Update documentation for API changes
- Test your changes thoroughly
- Keep the modular architecture intact

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
