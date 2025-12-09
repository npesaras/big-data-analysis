# Model Training Pipeline

This document explains the real-time training and evaluation process orchestrated by `src.pipeline`.

## Execution Flow

Upon user initiation, the system executes the following 7-step pipeline:

1. **Data Loading**: Ingestion of the PIMA Indians Diabetes dataset.
2. **Data Exploration**: Calculation of dataset statistics.
3. **Data Splitting**: Partitioning into training and testing sets based on user configuration.
4. **Preprocessing**: Application of cleaning, imputation, and scaling transformations.
5. **Model Training**: Parallel training of all 9 algorithms on the training set.
6. **Evaluation**: Assessment of each model on the testing set using accuracy, precision, recall, and F1-score.
7. **Prediction**: Generation of predictions for the specific patient data provided.

## Performance Metrics

Models are evaluated based on:

- **Accuracy**: Overall correctness of predictions.
- **Recall (Sensitivity)**: Ability to correctly identify positive diabetes cases.
- **Precision**: Accuracy of positive predictions.
- **F1-Score**: Harmonic mean of precision and recall.

The system automatically identifies the "Best Performing Model" based on the highest accuracy score.
