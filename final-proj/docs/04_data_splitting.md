# Data Splitting

This document details the methodology for partitioning data into training and testing sets.

## Methodology

The system employs a dynamic splitting strategy implemented in `src.pipeline`.

### Stratified Split

Data is split using Stratified Sampling to ensure that the class distribution (Diabetic vs. Non-Diabetic) remains consistent across both training and testing sets. This prevents bias in model evaluation.

### Configuration

The split ratio is user-configurable:

- **Training Set**: Adjustable from 60% to 90%.
- **Testing Set**: Automatically calculated as the remainder (10% to 40%).
- **Random Seed**: Configurable for reproducibility of splits.
