# Technical Documentation

Welcome to the technical documentation for the Diabetes Classification System. This documentation provides detailed explanations of each component and step in the machine learning pipeline.

## 📚 Documentation Structure

### Getting Started

- **[00. Installation Guide](00_installation.md)** - Setup instructions using uv package manager

### Data Pipeline

- **[01. Data Collection & Cleaning](01_data_collection.md)** - Loading and inspecting the diabetes dataset
- **[02. Exploratory Data Analysis](02_exploratory_analysis.md)** - Statistical analysis and visualizations
- **[03. Data Preprocessing](03_preprocessing.md)** - Handling missing values, scaling, and transformations
- **[04. Data Splitting](04_data_splitting.md)** - Train-test split and cross-validation strategies

### Model Development

- **[05. Model Selection](05_model_selection.md)** - Overview of 9 classification algorithms
- **[06. Model Training](06_model_training.md)** - Training procedures and cross-validation
- **[07. Model Evaluation](07_model_evaluation.md)** - Metrics, comparison, and selection criteria

### Reference

- **[08. API Reference](08_api_reference.md)** - Complete module and function documentation

## 🎯 Quick Navigation

### For Data Scientists

If you're interested in the data processing and feature engineering:
1. Start with [Data Collection](01_data_collection.md)
2. Explore [EDA](02_exploratory_analysis.md)
3. Learn about [Preprocessing](03_preprocessing.md)

### For ML Engineers

If you're interested in model development and deployment:
1. Review [Model Selection](05_model_selection.md)
2. Understand [Training](06_model_training.md)
3. Study [Evaluation](07_model_evaluation.md)

### For Developers

If you're contributing to the codebase:
1. Check [Installation](00_installation.md)
2. Reference the [API Documentation](08_api_reference.md)
3. Review the module structure in each guide

## 🏗️ Architecture Overview

The system follows a modular pipeline architecture:

```
Data → Cleaning → EDA → Preprocessing → Splitting → Training → Evaluation → Selection
```

Each step is implemented as a separate module in `src/`:

- **config.py** - Central configuration
- **data_cleaning.py** - Data loading and inspection
- **exploratory_data_analysis.py** - Statistical analysis and visualization
- **pre_processing.py** - Transformations and scaling
- **data_splitting.py** - Train-test split utilities
- **model_selection.py** - Model instantiation
- **model_training.py** - Training and cross-validation
- **model_evaluation.py** - Metrics and comparison
- **utils.py** - Helper functions

## 🎓 Learning Path

### Beginner Path

1. **Installation** → Setup environment
2. **Data Collection** → Understand the dataset
3. **EDA** → Visual exploration
4. **Run the App** → See it in action

### Intermediate Path

1. **Preprocessing** → Data transformations
2. **Data Splitting** → Validation strategies
3. **Model Training** → Train your first model

### Advanced Path

1. **Model Selection** → Compare algorithms
2. **Model Evaluation** → Deep dive into metrics
3. **API Reference** → Extend the system

## 🔍 Key Concepts

### Medical Context

This is a **medical diagnosis system** where:
- **False Negatives are costly** (missing diabetes diagnosis)
- **Recall is prioritized** over accuracy
- **Cost-sensitive evaluation** is applied

### Data Challenges

- **Missing data masked as zeros** in biological features
- **Class imbalance** (500 non-diabetic vs 268 diabetic)
- **Small sample size** (768 total samples)

### Technical Approach

- **Median imputation** for consistency
- **Stratified splitting** to maintain class balance
- **10-fold cross-validation** for robust evaluation
- **9 algorithm comparison** for best model selection

## 📊 Dataset Information

**PIMA Indians Diabetes Dataset**
- **Samples**: 768 patients
- **Features**: 8 clinical measurements
- **Target**: Binary (0=Non-Diabetic, 1=Diabetic)
- **Class Distribution**: 500:268 (1.87:1 ratio)

### Features

1. **Pregnancies** - Number of pregnancies
2. **Glucose** - Plasma glucose concentration
3. **BloodPressure** - Diastolic blood pressure
4. **SkinThickness** - Triceps skin fold thickness
5. **Insulin** - 2-hour serum insulin
6. **BMI** - Body mass index
7. **DiabetesPedigreeFunction** - Genetic influence score
8. **Age** - Age in years

## 🛠️ Technology Stack

- **Python 3.13+** - Programming language
- **scikit-learn** - ML algorithms and utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **plotly** - Interactive visualizations
- **seaborn** - Statistical plots
- **streamlit** - Web interface
- **uv** - Fast package management

## 📝 Contributing

When adding new features or documentation:

1. Follow the modular structure
2. Update relevant documentation
3. Add examples and code snippets
4. Include performance considerations
5. Document API changes

## 🔗 External Resources

- [PIMA Indians Dataset](https://archive.ics.uci.edu/dataset/34/diabetes) - Original data source
- [scikit-learn Documentation](https://scikit-learn.org/stable/) - ML library docs
- [Streamlit Documentation](https://docs.streamlit.io/) - Web app framework
- [uv Documentation](https://github.com/astral-sh/uv) - Package manager

## 📧 Support

For questions or issues:
- Check the relevant documentation section
- Review the API reference
- Open an issue on GitHub
- Contact the development team

---

**Last Updated**: December 9, 2025
**Version**: 2.0.0
