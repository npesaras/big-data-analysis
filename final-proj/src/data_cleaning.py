"""
Data loading and initial quality inspection functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_diabetes_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load the diabetes dataset from CSV file.

    Args:
        filepath: Path to the diabetes.csv file

    Returns:
        DataFrame containing the diabetes data, or None if loading fails
    """
    try:
        if not filepath.exists():
            logger.error(f"Data file not found: {filepath}")
            return None

        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None


def inspect_data(df: pd.DataFrame) -> Dict:
    """
    Perform initial data inspection and return summary statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing data inspection results
    """
    inspection = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'summary_stats': df.describe().to_dict()
    }

    logger.info(f"Data inspection complete - Shape: {inspection['shape']}, "
                f"Duplicates: {inspection['duplicates']}")

    return inspection


def identify_zero_issues(df: pd.DataFrame, cols_to_check: List[str]) -> Dict:
    """
    Identify zero values in columns where zeros are biologically impossible.

    Args:
        df: Input DataFrame
        cols_to_check: List of column names to check for zero issues

    Returns:
        Dictionary with zero value counts and percentages for each column
    """
    zero_issues = {}

    for col in cols_to_check:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            zero_issues[col] = {
                'count': int(zero_count),
                'percentage': round(zero_percentage, 2),
                'is_issue': zero_count > 0
            }

            if zero_count > 0:
                logger.warning(f"Column '{col}' has {zero_count} ({zero_percentage:.2f}%) zero values")

    return zero_issues


def get_data_quality_report(df: pd.DataFrame, target_column: str = 'Outcome') -> Dict:
    """
    Generate comprehensive data quality report.

    Args:
        df: Input DataFrame
        target_column: Name of the target variable column

    Returns:
        Dictionary containing comprehensive quality metrics
    """
    report = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,  # Exclude target
        'data_types': df.dtypes.value_counts().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }

    # Target variable analysis
    if target_column in df.columns:
        target_counts = df[target_column].value_counts().to_dict()
        report['target_distribution'] = target_counts
        report['class_balance_ratio'] = round(
            max(target_counts.values()) / min(target_counts.values()), 2
        )

    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_features'] = len(numeric_cols)
    report['feature_ranges'] = {}

    for col in numeric_cols:
        if col != target_column:
            report['feature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }

    logger.info(f"Data quality report generated for {report['total_samples']} samples")

    return report


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str],
                        multiplier: float = 1.5) -> Dict[str, pd.Series]:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Args:
        df: Input DataFrame
        columns: List of column names to check for outliers
        multiplier: IQR multiplier (default 1.5 for standard outliers)

    Returns:
        Dictionary mapping column names to boolean Series indicating outliers
    """
    outliers = {}

    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = outlier_mask

            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                logger.info(f"Column '{col}': {outlier_count} outliers detected "
                           f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")

    return outliers


def get_column_statistics(df: pd.DataFrame, column: str) -> Dict:
    """
    Get detailed statistics for a specific column.

    Args:
        df: Input DataFrame
        column: Name of the column

    Returns:
        Dictionary with detailed statistics
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return {}

    col_data = df[column]

    stats = {
        'count': int(col_data.count()),
        'missing': int(col_data.isnull().sum()),
        'unique': int(col_data.nunique()),
        'dtype': str(col_data.dtype)
    }

    if col_data.dtype in [np.int64, np.float64]:
        stats.update({
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'q25': float(col_data.quantile(0.25)),
            'q75': float(col_data.quantile(0.75)),
            'zeros': int((col_data == 0).sum())
        })

    return stats
