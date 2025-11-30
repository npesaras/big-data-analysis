# ITD105 – Big Data Analytics

## Lab Exercises #2

**Topic:** Utilizing Resampling Techniques and Performance Metrics for Classification, Regression, and Time Series Analysis

---

## Access our website here:

https://gumora-etal-itd105-lab2.streamlit.app/

## 📋 Project Overview

This lab exercise focuses on implementing and comparing different machine learning models using various resampling techniques and performance metrics. The project consists of two main parts: a classification task and a regression task.

---

## 📊 Part 1: Classification Task using K-Fold Cross-Validation

### Dataset Requirements

- **Type:** Health-related public dataset (classification type)
- **Suggested Sources:**
  - [Kaggle](https://www.kaggle.com/)
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
  - [Google Dataset Search](https://datasetsearch.research.google.com/)
  - Other public dataset repositories

### Model Architectures

| **Criteria**             | **Model A**               | **Model B**                    |
| ------------------------ | ------------------------- | ------------------------------ |
| **ML Algorithm**         | Logistic Regression       | Logistic Regression            |
| **Resampling Technique** | K-Fold Cross-Validation   | Leave-One-Out Cross Validation |
| **Performance Metrics**  | • Classification Accuracy | • Classification Accuracy      |
|                          | • Logarithmic Loss        | • Logarithmic Loss             |
|                          | • Confusion Matrix        | • Confusion Matrix             |
|                          | • Classification Report   | • Classification Report        |
|                          | • Area Under ROC Curve    | • Area Under ROC Curve         |

### Tasks

1. ✅ Load the Dataset
2. ✅ Build ML Models (Model A & Model B)
3. ✅ Interpret each performance metric
4. ✅ Select and justify either Model A or Model B
5. ✅ Download the selected model and integrate it into a Streamlit application for predictions

---

## 📈 Part 2: Regression Task using Train-Test Split

### Dataset Requirements

- **Type:** Environment-related public dataset (regression type)

### Model Architectures

| **Criteria**             | **Model A**                    | **Model B**                       |
| ------------------------ | ------------------------------ | --------------------------------- |
| **ML Algorithm**         | Linear Regression              | Linear Regression                 |
| **Resampling Technique** | Split into train and test sets | Repeated Random Train-Test Splits |
| **Performance Metrics**  | • Mean Squared Error (MSE)     | • Mean Squared Error (MSE)        |
|                          | • Mean Absolute Error (MAE)    | • Mean Absolute Error (MAE)       |
|                          | • R-squared (R²)               | • R-squared (R²)                  |

### Tasks

1. ✅ Load the Dataset
2. ✅ Build ML Models (Model A & Model B)
3. ✅ Interpret each performance metric
4. ✅ Select and justify either Model A or Model B
5. ✅ Download the selected model and integrate it into a Streamlit application for predictions

---

## 🎨 Streamlit Application Requirements

**Enhance the user-friendliness of your Streamlit app by:**

- Organizing layout with columns
- Using tabs for different sections
- Implementing various interactive widgets for filtering and data engagement
- Creating an intuitive interface for making predictions

---

## 📦 Submission Requirements

Please submit the following:

- ✅ **Source code** (.py file)
- ✅ **Datasets** used in the project
- ✅ **Video Screen Recording** of your output
  - Maximum duration: **5 minutes**
  - Content: Explain the features and show sample predictions

---

## 📚 Key Concepts

### Classification Metrics

- **Classification Accuracy:** Percentage of correctly classified instances
- **Logarithmic Loss:** Measures the uncertainty of predictions
- **Confusion Matrix:** Visualizes true vs predicted classifications
- **Classification Report:** Precision, recall, F1-score for each class
- **Area Under ROC Curve:** Model's ability to distinguish between classes

### Regression Metrics

- **Mean Squared Error (MSE):** Average squared difference between predicted and actual values
- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values
- **R-squared (R²):** Proportion of variance in the dependent variable explained by the model

### Resampling Techniques

- **K-Fold Cross-Validation:** Dataset divided into k subsets, model trained k times
- **Leave-One-Out Cross Validation:** Each observation used once as validation set
- **Train-Test Split:** Simple division of data into training and testing sets
- **Repeated Random Train-Test Splits:** Multiple random splits to assess model stability

---

## 🚀 Getting Started

1. Select appropriate datasets for classification and regression tasks
2. Implement both models (A & B) for each task
3. Evaluate and compare performance metrics
4. Choose the best-performing model with justification
5. Develop an interactive Streamlit application
6. Record a demonstration video

---

## 📝 Notes

- Ensure datasets are properly preprocessed before model training
- Document your reasoning for model selection
- Make the Streamlit application user-friendly and interactive
- Test predictions with various inputs to demonstrate model capabilities

---

**Course:** ITD105 – Big Data Analytics  
**Lab:** Exercise #2
