# Big Data Analytics Course Projects

A comprehensive collection of data analytics projects demonstrating advanced machine learning techniques, interactive data visualization, and modern Python development practices.

## Table of Contents

- [Overview](#overview)
- [Labs](#labs)
- [Technologies](#technologies)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Course Information](#course-information)
- [Author](#author)
- [License](#license)

---

## Overview

This repository contains three major projects for the ITD105 Big Data Analytics course, showcasing different aspects of data science and machine learning:

- **Lab 1**: Interactive Exploratory Data Analysis (EDA) dashboard
- **Lab 2**: Advanced machine learning models with modular architecture
- **Final Project**: Real-Time Dynamic ML Training System for diabetes classification

Each project demonstrates industry-standard practices including modern Python packaging, comprehensive documentation, and production-ready code structure.

---

## Labs

### Lab 1: Student Performance Analysis Dashboard

**Location**: [`lab1/`](lab1/)

A comprehensive Exploratory Data Analysis (EDA) dashboard for student exam performance built with Streamlit. This interactive application provides:

- **Complete EDA Suite**: Dataset overview, statistical summaries, and correlation analysis
- **Advanced Visualizations**: Heatmaps, boxplots, scatter plots, and pair plots
- **Automated Lab Analysis**: Addresses specific course requirements with data-driven insights
- **Smart Filtering**: Dynamic filtering by gender, age, and parental education
- **Responsive Design**: Optimized for both desktop and mobile devices

**Key Features**:

- Interactive correlation heatmaps
- Automated analysis of lab questions
- Gender-based performance comparisons
- Study time impact analysis
- Real-time data filtering and visualization

### Lab 2: Machine Learning Models Suite

**Location**: [`lab2/`](lab2/)

An advanced implementation of classification and regression models featuring a modular architecture that separates data processing from model training for better maintainability and scalability.

**Machine Learning Tasks**:

- **Diabetes Classification**: Binary classification using PIMA Indians Diabetes Dataset
- **Housing Price Regression**: Regression analysis using Boston Housing Dataset

**Advanced Features**:

- **Modular Architecture**: Complete separation of data cleaning and model training
- **Advanced Resampling**: K-Fold, Leave-One-Out, and Repeated K-Fold Cross-Validation
- **Comprehensive Metrics**: Accuracy, Log Loss, ROC AUC, Confusion Matrix, MAE, MSE, RMSE, R²
- **Web Interface**: Interactive Streamlit application for real-time predictions
- **Configuration Management**: Centralized path and parameter management

### Final Project: Real-Time Dynamic ML Training System

**Location**: [`final-proj/`](final-proj/)

An advanced real-time machine learning system that trains 9 classification algorithms dynamically based on user-configured parameters. This production-ready application features a unified single-page interface for diabetes prediction.

**System Capabilities**:

- **Real-Time Training**: Train all 9 ML algorithms from scratch in 10-30 seconds
- **Dynamic Configuration**: Adjust train/test split, K-neighbors, imputation strategy, and preprocessing options
- **9 ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, KNN, SVM, AdaBoost, Perceptron, MLP Neural Network
- **Comprehensive Pipeline**: Complete workflow from data loading to prediction with progress tracking
- **Interactive Interface**: Single-page application with sidebar configuration and patient input form

**Key Features**:

- Configurable training parameters (train/test split, random seed, K-neighbors)
- Real-time preprocessing with zero-value handling and imputation
- Immediate model comparison with sortable performance metrics
- Patient-specific predictions with confidence scores and clinical interpretation
- Visual analytics including probability charts and accuracy comparisons
- Model consensus display showing agreement across all 9 algorithms
- Download options for results and comparison tables

**Advanced Architecture**:

- Modular pipeline orchestration with `src/pipeline.py`
- Enhanced model evaluation with prediction comparison functions
- Dynamic KNN parameter configuration
- Comprehensive error handling and progress tracking
- Clinical risk assessment and recommendations

---

## Technologies

### Core Technologies

- **Python 3.13** - Primary programming language
- **uv** - Modern Python package manager
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Visualization & Analysis

- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical data visualization
- **joblib** - Model serialization

### Development Tools

- **mypy** - Type checking
- **flake8** - Linting
- **black** - Code formatting
- **isort** - Import sorting

---

## Installation

### Prerequisites

- **Python**: 3.12 or higher
- **Package Manager**: uv (recommended) or pip
- **Git**: For cloning the repository

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/npesaras/big-data-analysis.git
   cd big-data-analysis
   ```

2. **Install uv package manager** (if not already installed)

   **On macOS/Linux:**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **On Windows:**

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
   ```

3. **Navigate to each project and install dependencies**

   **For Lab 1:**

   ```bash
   cd lab1
   uv sync
   ```

   **For Lab 2:**

   ```bash
   cd ../lab2
   uv sync
   ```

   **For Final Project:**

   ```bash
   cd ../final-proj
   uv sync
   ```

### Dataset Setup

**Lab 1 Dataset:**

- Download `student-mat.csv` from: [Google Drive Link](https://drive.google.com/drive/folders/1Bz9q37BB20PJSWsdGH__cshZGfPKSpHd?usp=sharing)
- Place the file in `lab1/data/` directory

**Lab 2 Datasets:**

- Pre-included: `diabetes.csv` and `house-data.csv` in `lab2/data/`

**Final Project Dataset:**

- Pre-included: `diabetes.csv` (PIMA Indians Diabetes Dataset) in `final-proj/data/`

### Running the Applications

**Lab 1 - Student Performance Dashboard:**

```bash
cd lab1
uv run streamlit run main.py
```

**Lab 2 - ML Models Suite:**

```bash
cd lab2
uv run streamlit run main.py
```

**Final Project - Real-Time ML Training System:**

```bash
cd final-proj
uv run streamlit run main.py
```

---

## Project Structure

```text
big-data-analysis/
├── lab1/                          # Lab Exercise #1
│   ├── main.py                    # Streamlit EDA dashboard
│   ├── pyproject.toml            # Project configuration
│   ├── requirements.txt          # Dependencies
│   ├── README.md                 # Lab 1 documentation
│   ├── data/                     # Dataset directory
│   │   └── student-mat.csv       # Student performance data
│   ├── docs/                     # Documentation
│   └── assets/                   # Static assets
│
├── lab2/                         # Lab Exercise #2
│   ├── main.py                   # Streamlit ML application
│   ├── pyproject.toml           # Project configuration
│   ├── README.md                # Lab 2 documentation
│   ├── regression.py            # Legacy implementation
│   ├── src/                     # Core business logic
│   │   ├── __init__.py          # Package metadata
│   │   ├── config.py            # Centralized configuration
│   │   ├── data_cleaning.py     # Data loading & preprocessing
│   │   ├── models.py            # ML model creation & evaluation
│   │   └── utils.py             # Helper functions
│   ├── scripts/                 # Training orchestration
│   │   ├── train_classification.py
│   │   └── train_regression.py
│   ├── models/                  # Trained model artifacts
│   ├── data/                    # Raw datasets
│   └── docs/                    # Documentation
│
├── final-proj/                  # Final Project
│   ├── main.py                  # Real-time ML training system
│   ├── pyproject.toml          # Project configuration
│   ├── README.md               # Final project documentation
│   ├── src/                    # Core modules
│   │   ├── __init__.py         # Package initialization
│   │   ├── config.py           # Configuration and hyperparameters
│   │   ├── pipeline.py         # Complete ML pipeline orchestration
│   │   ├── data_cleaning.py    # Data loading and inspection
│   │   ├── pre_processing.py   # Preprocessing pipeline
│   │   ├── model_selection.py  # 9 algorithm definitions
│   │   ├── model_training.py   # Training procedures
│   │   ├── model_evaluation.py # Metrics and comparison
│   │   ├── data_splitting.py   # Train/test split utilities
│   │   ├── exploratory_data_analysis.py  # Visualizations
│   │   └── utils.py            # Helper functions
│   ├── models/                 # Model artifacts (generated)
│   ├── data/                   # PIMA Indians Diabetes dataset
│   └── docs/                   # Technical documentation
│
└── README.md                    # This file
```

---

### Course Projects

1. **Lab 1**: Student Performance Analysis
   - **Objective**: Build an interactive EDA dashboard using Streamlit
   - **Focus**: Data exploration, visualization, and statistical analysis
   - **Skills**: Pandas, Plotly, statistical analysis, interactive dashboards

2. **Lab 2**: Machine Learning Implementation
   - **Objective**: Implement classification and regression models with advanced techniques
   - **Focus**: Modular architecture, cross-validation, and model evaluation
   - **Skills**: scikit-learn, model selection, hyperparameter tuning, CV techniques

3. **Final Project**: Real-Time Dynamic ML Training System
   - **Objective**: Build a production-ready ML system with real-time training capabilities
   - **Focus**: Pipeline orchestration, dynamic configuration, clinical decision support
   - **Skills**: Advanced ML pipelines, real-time processing, model comparison, production deployment

---

## License

This project is developed as part of an academic course assignment. All rights reserved to the author and educational institution.
