"""
Exploratory Data Analysis with interactive Plotly visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
import logging

logger = logging.getLogger(__name__)


def plot_target_distribution(df: pd.DataFrame, target_column: str = 'Outcome'):
    """
    Create an interactive bar chart showing target variable distribution.

    Args:
        df: Input DataFrame
        target_column: Name of the target variable

    Returns:
        Plotly figure object
    """
    target_counts = df[target_column].value_counts().sort_index()

    fig = go.Figure(data=[
        go.Bar(
            x=['Non-Diabetic (0)', 'Diabetic (1)'],
            y=target_counts.values,
            text=target_counts.values,
            textposition='auto',
            marker_color=['#2ecc71', '#e74c3c']
        )
    ])

    fig.update_layout(
        title='Target Variable Distribution (Class Balance)',
        xaxis_title='Class',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )

    logger.info("Target distribution plot created")
    return fig


def plot_feature_distributions(df: pd.DataFrame, feature_columns: List[str],
                               target_column: str = 'Outcome'):
    """
    Create histograms for all features colored by target variable.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of the target variable

    Returns:
        Plotly figure object
    """
    n_cols = 3
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=feature_columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, col in enumerate(feature_columns):
        row = idx // n_cols + 1
        col_num = idx % n_cols + 1

        for outcome in [0, 1]:
            data = df[df[target_column] == outcome][col]
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f'Class {outcome}',
                    legendgroup=f'class{outcome}',
                    showlegend=(idx == 0),
                    marker_color='#2ecc71' if outcome == 0 else '#e74c3c',
                    opacity=0.7
                ),
                row=row,
                col=col_num
            )

    fig.update_layout(
        title_text='Feature Distributions by Target Class',
        height=300 * n_rows,
        showlegend=True,
        barmode='overlay'
    )

    logger.info("Feature distribution plots created")
    return fig


def plot_box_plots(df: pd.DataFrame, feature_columns: List[str],
                   target_column: str = 'Outcome'):
    """
    Create box plots for all features grouped by target variable.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of the target variable

    Returns:
        Plotly figure object
    """
    n_cols = 3
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=feature_columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, col in enumerate(feature_columns):
        row = idx // n_cols + 1
        col_num = idx % n_cols + 1

        for outcome in [0, 1]:
            data = df[df[target_column] == outcome][col]
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f'Class {outcome}',
                    legendgroup=f'class{outcome}',
                    showlegend=(idx == 0),
                    marker_color='#2ecc71' if outcome == 0 else '#e74c3c'
                ),
                row=row,
                col=col_num
            )

    fig.update_layout(
        title_text='Feature Box Plots by Target Class',
        height=300 * n_rows,
        showlegend=True
    )

    logger.info("Box plots created")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, feature_columns: List[str]):
    """
    Create an interactive correlation heatmap.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names (including target)

    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = df[feature_columns].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        width=700,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    logger.info("Correlation matrix plot created")
    return fig


def identify_predictive_features(df: pd.DataFrame, feature_columns: List[str],
                                 target_column: str = 'Outcome') -> pd.DataFrame:
    """
    Calculate correlation of each feature with the target variable.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of the target variable

    Returns:
        DataFrame with features sorted by absolute correlation with target
    """
    correlations = {}

    for col in feature_columns:
        corr = df[col].corr(df[target_column])
        correlations[col] = corr

    corr_df = pd.DataFrame({
        'Feature': list(correlations.keys()),
        'Correlation': list(correlations.values()),
        'Abs_Correlation': [abs(v) for v in correlations.values()]
    })

    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

    logger.info("Feature correlations calculated")
    return corr_df


def plot_feature_importance_correlation(df: pd.DataFrame, feature_columns: List[str],
                                       target_column: str = 'Outcome'):
    """
    Create a bar chart showing feature correlations with target variable.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of the target variable

    Returns:
        Plotly figure object
    """
    corr_df = identify_predictive_features(df, feature_columns, target_column)

    # Create color based on positive/negative correlation
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr_df['Correlation']]

    fig = go.Figure(data=[
        go.Bar(
            x=corr_df['Correlation'],
            y=corr_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=np.round(corr_df['Correlation'], 3),
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Feature Correlation with Target Variable',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Features',
        height=400,
        showlegend=False
    )

    logger.info("Feature importance correlation plot created")
    return fig


def plot_pairwise_relationships(df: pd.DataFrame, features: List[str],
                                target_column: str = 'Outcome'):
    """
    Create scatter plot matrix for selected features.

    Args:
        df: Input DataFrame
        features: List of feature columns to plot (max 4-5 recommended)
        target_column: Name of the target variable

    Returns:
        Plotly figure object
    """
    # Add target to the dataframe for coloring
    plot_df = df[features + [target_column]].copy()
    plot_df[target_column] = plot_df[target_column].astype(str)

    fig = px.scatter_matrix(
        plot_df,
        dimensions=features,
        color=target_column,
        color_discrete_map={'0': '#2ecc71', '1': '#e74c3c'},
        labels={target_column: 'Outcome'},
        height=800
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(title='Pairwise Feature Relationships')

    logger.info("Pairwise relationship plot created")
    return fig


def create_summary_statistics_table(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Create a comprehensive summary statistics table.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names

    Returns:
        DataFrame with summary statistics
    """
    summary = df[feature_columns].describe().T
    summary['zeros'] = df[feature_columns].apply(lambda x: (x == 0).sum())
    summary['zeros_pct'] = (summary['zeros'] / len(df) * 100).round(2)
    summary['missing'] = df[feature_columns].isnull().sum()
    summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(2)

    # Reorder columns
    summary = summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                       'zeros', 'zeros_pct', 'missing', 'missing_pct']]

    logger.info("Summary statistics table created")
    return summary
