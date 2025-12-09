# Data Collection

This document details the dataset used in the machine learning pipeline.

## Dataset Specification

The system utilizes the PIMA Indians Diabetes Database, sourced from the National Institute of Diabetes and Digestive and Kidney Diseases.

### Characteristics

- **Subject Count**: 768 patients
- **Feature Count**: 8 clinical measurements
- **Target Variable**: Binary Outcome (0: Negative, 1: Positive)
- **Demographics**: Female patients of Pima Indian heritage, 21 years or older

### Feature Definitions

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg / height in m^2)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic score)
8. **Age**: Age in years

### Data Integrity

The dataset contains missing values encoded as zeros in biologically impossible fields (e.g., Glucose, BloodPressure, BMI). These are handled dynamically during the preprocessing stage.
