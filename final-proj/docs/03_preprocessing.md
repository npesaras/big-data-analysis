# Data Preprocessing

This document explains the data transformation pipeline implemented in `src.pre_processing`.

## Pipeline Architecture

The preprocessing pipeline executes the following transformations in sequence:

1. **Zero Handling**: Identification and replacement of zero values in biologically impossible fields (Glucose, BloodPressure, SkinThickness, Insulin, BMI) with NaN.
2. **Imputation**: Filling of missing values using statistical strategies (Mean or Median), configurable by the user.
3. **Scaling**: Standardization of features using `StandardScaler` to achieve zero mean and unit variance.

## Configuration

The preprocessing parameters are dynamically configurable via the user interface:

- **Imputation Strategy**: Selection between Mean and Median.
- **Zero Handling**: Toggle to enable or disable the treatment of zeros as missing values.
