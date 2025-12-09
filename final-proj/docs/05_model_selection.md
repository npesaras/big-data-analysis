# Model Selection

This document lists the classification algorithms utilized by the system.

## Algorithm Suite

The system trains and evaluates 9 distinct machine learning algorithms simultaneously:

1.  **Logistic Regression**: A linear model for binary classification.
2.  **Decision Tree**: A non-parametric supervised learning method.
3.  **Random Forest**: An ensemble learning method constructing multiple decision trees.
4.  **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional spaces.
5.  **K-Nearest Neighbors (KNN)**: An instance-based learning algorithm (K is configurable).
6.  **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
7.  **AdaBoost**: An ensemble boosting classifier.
8.  **Gradient Boosting**: An ensemble technique that builds models sequentially.
9.  **Extra Trees**: An extremely randomized tree ensemble method.

## Configuration

- **KNN Neighbors**: The 'K' parameter for the K-Nearest Neighbors algorithm is adjustable via the interface.
