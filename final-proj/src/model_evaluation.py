"""
Model evaluation functions with comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model: Any,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  model_name: str = "Model") -> Dict:
    """
    Evaluate a trained model on test data with comprehensive metrics.

    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'support': len(y_test)
        }

        # Add ROC-AUC if probability predictions available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Calculate specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.info(f"Evaluated {model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        return {'model_name': model_name, 'error': str(e)}


def evaluate_all_models(models: Dict[str, Any],
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Dict]:
    """
    Evaluate all trained models on test data.

    Args:
        models: Dictionary of trained model instances
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with evaluation results for all models
    """
    results = {}
    total_models = len(models)

    logger.info(f"Evaluating {total_models} models on test set...")

    for idx, (model_name, model) in enumerate(models.items(), 1):
        logger.info(f"[{idx}/{total_models}] Evaluating {model_name}...")
        results[model_name] = evaluate_model(model, X_test, y_test, model_name)

    logger.info(f"All {total_models} models evaluated successfully")

    return results


def create_comparison_table(eval_results: Dict[str, Dict],
                           sort_by: str = 'recall') -> pd.DataFrame:
    """
    Create a comparison table of all model evaluation results.

    Args:
        eval_results: Dictionary with evaluation results for all models
        sort_by: Metric to sort by (default: recall for medical diagnosis)

    Returns:
        DataFrame with model comparison
    """
    comparison_data = []

    for model_name, results in eval_results.items():
        if 'error' in results:
            continue

        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'ROC-AUC': results.get('roc_auc', 0),
            'Specificity': results.get('specificity', 0),
            'TP': results.get('true_positives', 0),
            'TN': results.get('true_negatives', 0),
            'FP': results.get('false_positives', 0),
            'FN': results.get('false_negatives', 0)
        }
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Sort by specified metric
    if sort_by in df_comparison.columns:
        df_comparison = df_comparison.sort_values(sort_by, ascending=False)
    elif sort_by.capitalize() in df_comparison.columns:
        df_comparison = df_comparison.sort_values(sort_by.capitalize(), ascending=False)

    logger.info(f"Comparison table created - Best {sort_by}: {df_comparison.iloc[0]['Model']} "
                f"({df_comparison.iloc[0][sort_by.capitalize() if sort_by.capitalize() in df_comparison.columns else sort_by]:.4f})")

    return df_comparison


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metrics: List[str] = None):
    """
    Create an interactive bar chart comparing models across metrics.

    Args:
        comparison_df: DataFrame with model comparison
        metrics: List of metrics to plot

    Returns:
        Plotly figure object
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    fig = go.Figure()

    for metric in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(4),
                textposition='auto'
            ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        showlegend=True,
        yaxis_range=[0, 1]
    )

    logger.info("Model comparison plot created")
    return fig


def plot_confusion_matrices(eval_results: Dict[str, Dict]):
    """
    Create a grid of confusion matrices for all models.

    Args:
        eval_results: Dictionary with evaluation results containing confusion matrices

    Returns:
        Plotly figure object
    """
    models = [name for name, res in eval_results.items() if 'confusion_matrix' in res]
    n_models = len(models)

    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=models,
        specs=[[{'type': 'heatmap'} for _ in range(n_cols)] for _ in range(n_rows)]
    )

    for idx, model_name in enumerate(models):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        cm = np.array(eval_results[model_name]['confusion_matrix'])

        # Create annotations with counts
        annotations_text = [[f'TN<br>{cm[0,0]}', f'FP<br>{cm[0,1]}'],
                           [f'FN<br>{cm[1,0]}', f'TP<br>{cm[1,1]}']]

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                showscale=(idx == 0),
                text=annotations_text,
                texttemplate='%{text}',
                textfont={"size": 12}
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title_text='Confusion Matrices for All Models',
        height=300 * n_rows,
        showlegend=False
    )

    logger.info("Confusion matrices plot created")
    return fig


def plot_roc_curves(models: Dict[str, Any],
                   X_test: pd.DataFrame,
                   y_test: pd.Series):
    """
    Plot ROC curves for all models that support probability predictions.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC={auc_score:.3f})',
                    line=dict(width=2)
                ))
            except Exception as e:
                logger.warning(f"Could not plot ROC for {model_name}: {str(e)}")

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray', width=1)
    ))

    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        height=500,
        showlegend=True
    )

    logger.info("ROC curves plot created")
    return fig


def get_best_model(eval_results: Dict[str, Dict],
                  metric: str = 'recall') -> Tuple[str, Dict]:
    """
    Get the best performing model based on specified metric.

    Args:
        eval_results: Dictionary with evaluation results
        metric: Metric to use for selection (default: recall for medical diagnosis)

    Returns:
        Tuple of (best model name, its evaluation results)
    """
    valid_results = {name: res for name, res in eval_results.items()
                    if 'error' not in res and metric in res}

    if not valid_results:
        logger.error(f"No valid results found for metric '{metric}'")
        return None, None

    best_model_name = max(valid_results, key=lambda x: valid_results[x][metric])
    best_results = valid_results[best_model_name]

    logger.info(f"Best model by {metric}: {best_model_name} ({best_results[metric]:.4f})")

    return best_model_name, best_results


def generate_classification_report_dict(y_test: pd.Series,
                                        y_pred: np.ndarray) -> Dict:
    """
    Generate detailed classification report as dictionary.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with classification report
    """

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Classification report generated")
    return report_dict


def calculate_cost_sensitive_metrics(eval_results: Dict[str, Dict],
                                     fn_cost: float = 5.0,
                                     fp_cost: float = 1.0) -> pd.DataFrame:
    """
    Calculate cost-sensitive evaluation considering medical diagnosis costs.
    False negatives (missing diabetes) are more costly than false positives.

    Args:
        eval_results: Dictionary with evaluation results
        fn_cost: Cost of false negative (default: 5x)
        fp_cost: Cost of false positive (default: 1x)

    Returns:
        DataFrame with cost analysis
    """
    cost_data = []

    for model_name, results in eval_results.items():
        if 'error' in results:
            continue

        fn = results.get('false_negatives', 0)
        fp = results.get('false_positives', 0)

        total_cost = (fn * fn_cost) + (fp * fp_cost)

        cost_data.append({
            'Model': model_name,
            'False_Negatives': fn,
            'False_Positives': fp,
            'FN_Cost': fn * fn_cost,
            'FP_Cost': fp * fp_cost,
            'Total_Cost': total_cost,
            'Recall': results.get('recall', 0)
        })

    df_cost = pd.DataFrame(cost_data)
    df_cost = df_cost.sort_values('Total_Cost', ascending=True)

    logger.info(f"Cost-sensitive analysis completed - "
                f"Lowest cost model: {df_cost.iloc[0]['Model']} "
                f"(Cost: {df_cost.iloc[0]['Total_Cost']:.2f})")

    return df_cost


def get_evaluation_summary(eval_results: Dict[str, Dict]) -> Dict:
    """
    Generate summary statistics from evaluation results.

    Args:
        eval_results: Dictionary with evaluation results

    Returns:
        Dictionary with summary statistics
    """
    valid_results = [r for r in eval_results.values() if 'error' not in r]

    if not valid_results:
        return {'error': 'No valid evaluation results'}

    summary = {
        'total_models': len(eval_results),
        'successful_evaluations': len(valid_results),
        'avg_accuracy': np.mean([r['accuracy'] for r in valid_results]),
        'avg_recall': np.mean([r['recall'] for r in valid_results]),
        'avg_precision': np.mean([r['precision'] for r in valid_results]),
        'avg_f1': np.mean([r['f1_score'] for r in valid_results]),
        'best_accuracy_model': max(valid_results, key=lambda x: x['accuracy'])['model_name'],
        'best_recall_model': max(valid_results, key=lambda x: x['recall'])['model_name'],
        'best_f1_model': max(valid_results, key=lambda x: x['f1_score'])['model_name']
    }

    logger.info(f"Evaluation summary generated - Avg Accuracy: {summary['avg_accuracy']:.4f}, "
                f"Avg Recall: {summary['avg_recall']:.4f}")

    return summary
