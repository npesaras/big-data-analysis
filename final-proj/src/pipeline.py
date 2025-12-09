"""
Complete ML pipeline for real-time training and prediction.
Orchestrates data loading, preprocessing, training, evaluation, and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, Any
import logging
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

from src.data_cleaning import load_diabetes_data
from src.pre_processing import preprocess_pipeline, apply_preprocessing
from src.model_selection import create_all_models
from src.model_evaluation import evaluate_all_models, create_comparison_table
import src.config as config

logger = logging.getLogger(__name__)


def run_full_pipeline(
    patient_data: Dict[str, float],
    train_size: float,
    k_neighbors: int,
    random_seed: int,
    imputation_strategy: str,
    handle_zeros: bool
) -> Dict[str, Any]:
    """
    Execute complete ML pipeline in real-time:
    Load → Clean → Split → Preprocess → Train 9 models → Evaluate → Predict

    Args:
        patient_data: Dictionary with patient's diagnostic measurements
        train_size: Fraction of data for training (0.0-1.0)
        k_neighbors: Number of neighbors for KNN algorithm
        random_seed: Random state for reproducibility
        imputation_strategy: Strategy for imputation ('median' or 'mean')
        handle_zeros: Whether to replace zeros with NaN

    Returns:
        Dictionary containing all results (models, predictions, metrics, etc.)
    """

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = {
        'success': False,
        'error': None,
        'trained_models': None,
        'eval_results': None,
        'patient_predictions': None,
        'best_model_name': None,
        'dataset_info': None,
        'X_test': None,
        'y_test': None
    }

    try:
        # ========= STEP 1: LOAD DATA =========
        status_text.text('Step 1/7: Loading diabetes dataset...')
        df = load_diabetes_data(config.DIABETES_DATA)

        if df is None:
            raise ValueError("Failed to load dataset")

        total_samples = len(df)
        train_samples = int(total_samples * train_size)
        test_samples = total_samples - train_samples

        results['dataset_info'] = {
            'total': total_samples,
            'train': train_samples,
            'test': test_samples,
            'train_pct': train_size * 100,
            'test_pct': (1 - train_size) * 100
        }

        progress_bar.progress(10)
        time.sleep(0.3)

        # ========= STEP 2: DATA EXPLORATION =========
        status_text.text('Step 2/7: Exploring data...')
        st.caption(f"✓ Loaded {total_samples} patient records")
        st.caption(f"✓ Features: {len(config.FEATURE_COLUMNS)}, Target: Outcome (0/1)")
        st.caption(f"✓ Diabetic cases: {df[config.TARGET_COLUMN].sum()}, Healthy: {(df[config.TARGET_COLUMN] == 0).sum()}")

        progress_bar.progress(20)
        time.sleep(0.3)

        # ========= STEP 3: SPLIT DATA =========
        status_text.text('Step 3/7: Splitting data into train/test sets...')
        X = df[config.FEATURE_COLUMNS]
        y = df[config.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=(1 - train_size),
            random_state=random_seed,
            stratify=y
        )

        st.caption(f"✓ Training set: {len(X_train)} samples ({train_size*100:.0f}%)")
        st.caption(f"✓ Test set: {len(X_test)} samples ({(1-train_size)*100:.0f}%)")

        progress_bar.progress(30)
        time.sleep(0.3)

        # ========= STEP 4: PREPROCESSING =========
        status_text.text('Step 4/7: Preprocessing (handling zeros, imputation, scaling)...')

        # Prepare training data for preprocessing
        train_df = X_train.copy()
        train_df[config.TARGET_COLUMN] = y_train

        # Determine which columns to fix
        cols_to_fix = config.COLS_WITH_ZERO_ISSUES if handle_zeros else []

        # Apply preprocessing pipeline
        X_train_processed, _, imputer, scaler = preprocess_pipeline(
            train_df,
            config.FEATURE_COLUMNS,
            config.TARGET_COLUMN,
            cols_to_fix,
            imputation_strategy
        )

        # Apply same preprocessing to test data
        X_test_processed = apply_preprocessing(
            X_test,
            cols_to_fix,
            imputer,
            scaler
        )

        st.caption(f"✓ Imputation: {imputation_strategy}")
        st.caption(f"✓ Scaling: StandardScaler (mean=0, std=1)")
        st.caption(f"✓ Zero handling: {'Enabled' if handle_zeros else 'Disabled'}")

        progress_bar.progress(45)
        time.sleep(0.3)

        # ========= STEP 5: MODEL TRAINING =========
        status_text.text('Step 5/7: Training all 9 machine learning algorithms...')

        # Create models with dynamic K-neighbors
        models = create_all_models(config.MODELS_CONFIG, k_neighbors=k_neighbors)

        # Train all models on full training set
        trained_models = {}
        model_names = list(models.keys())

        for idx, (name, model) in enumerate(models.items()):
            # Train each model
            model.fit(X_train_processed, y_train)
            trained_models[name] = model

            # Update progress
            train_progress = 45 + int((idx + 1) / len(models) * 25)
            progress_bar.progress(train_progress)

        st.caption(f"✓ Trained {len(trained_models)} models successfully")
        st.caption(f"✓ KNN neighbors: {k_neighbors}")

        progress_bar.progress(70)
        time.sleep(0.3)

        # ========= STEP 6: EVALUATION =========
        status_text.text('Step 6/7: Evaluating models on test set...')

        # Evaluate all models
        eval_results = evaluate_all_models(
            trained_models,
            X_test_processed,
            y_test
        )

        # Find best model by accuracy
        comparison_df = create_comparison_table(eval_results, sort_by='accuracy')
        best_model_name = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']

        st.caption(f"✓ Best model: {best_model_name} (Accuracy: {best_accuracy:.2%})")

        progress_bar.progress(85)
        time.sleep(0.3)

        # ========= STEP 7: PREDICTION =========
        status_text.text('Step 7/7: Making predictions for patient data...')

        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])

        # Preprocess patient data
        patient_processed = apply_preprocessing(
            patient_df,
            cols_to_fix,
            imputer,
            scaler
        )

        # Get predictions from all models
        patient_predictions = {}
        for model_name, model in trained_models.items():
            pred = model.predict(patient_processed)[0]

            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(patient_processed)[0]
                diabetes_prob = proba[1] * 100
                confidence = max(proba) * 100
            else:
                diabetes_prob = None
                confidence = None

            patient_predictions[model_name] = {
                'prediction': int(pred),
                'prediction_label': 'Diabetic' if pred == 1 else 'Not Diabetic',
                'diabetes_probability': diabetes_prob,
                'confidence': confidence
            }

        st.caption(f"✓ Generated predictions from all {len(trained_models)} models")

        progress_bar.progress(100)
        time.sleep(0.5)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Store results
        results['success'] = True
        results['trained_models'] = trained_models
        results['eval_results'] = eval_results
        results['patient_predictions'] = patient_predictions
        results['best_model_name'] = best_model_name
        results['X_test'] = X_test_processed
        results['y_test'] = y_test

        logger.info(f"Pipeline completed successfully. Best model: {best_model_name}")

        return results

    except Exception as e:
        progress_bar.empty()
        status_text.empty()

        error_msg = f"Pipeline execution failed: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")

        results['error'] = str(e)
        return results


def display_results(
    results: Dict[str, Any],
    patient_data: Dict[str, float],
    train_size: float
):
    """
    Display comprehensive results after pipeline execution.

    Args:
        results: Dictionary containing all pipeline results
        patient_data: Original patient input data
        train_size: Training data fraction used
    """

    if not results['success']:
        st.error("❌ Pipeline execution failed. Please check the error message above.")
        return

    # Extract results
    eval_results = results['eval_results']
    patient_predictions = results['patient_predictions']
    best_model_name = results['best_model_name']
    dataset_info = results['dataset_info']
    trained_models = results['trained_models']
    X_test = results['X_test']
    y_test = results['y_test']

    # ========= PREDICTION FOR THIS PATIENT =========
    st.markdown("---")
    st.subheader("Prediction for This Patient")

    best_pred = patient_predictions[best_model_name]

    if best_pred['prediction'] == 1:
        st.error(f"**Prediction: {best_pred['prediction_label']}**")
    else:
        st.success(f"**Prediction: {best_pred['prediction_label']}**")

    if best_pred['confidence'] is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Confidence", f"{best_pred['confidence']:.1f}%")
        col_b.metric("Diabetes Probability", f"{best_pred['diabetes_probability']:.1f}%")
        col_c.metric("Healthy Probability", f"{100 - best_pred['diabetes_probability']:.1f}%")

    # ========= BEST MODEL SECTION =========
    st.markdown("---")
    st.subheader("Best Performing Model")

    best_model_metrics = eval_results[best_model_name]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithm", best_model_name)
    with col2:
        st.metric("Test Accuracy", f"{best_model_metrics['accuracy']:.2%}")
    with col3:
        st.metric("Recall", f"{best_model_metrics['recall']:.2%}")
    with col4:
        st.metric("Precision", f"{best_model_metrics['precision']:.2%}")

    st.info(f"Trained on {dataset_info['train']} samples ({dataset_info['train_pct']:.0f}%), "
            f"tested on {dataset_info['test']} samples ({dataset_info['test_pct']:.0f}%)")

    # ========= ALL MODELS COMPARISON =========
    st.markdown("---")
    st.subheader("All 9 Algorithms Comparison")

    # Create prediction comparison DataFrame
    from src.model_evaluation import create_prediction_comparison_df

    pred_comparison_df = create_prediction_comparison_df(
        eval_results,
        patient_predictions
    )

    # Display sortable table
    st.dataframe(
        pred_comparison_df.style.format({
            'Test Accuracy': '{:.2%}',
            'Diabetes Prob': '{:.1f}%',
            'Confidence': '{:.1f}%'
        }).background_gradient(
            subset=['Test Accuracy'],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )

    # Model consensus
    diabetic_count = sum(1 for p in patient_predictions.values() if p['prediction'] == 1)
    healthy_count = len(patient_predictions) - diabetic_count

    col_consensus1, col_consensus2 = st.columns(2)
    col_consensus1.metric("Models Predict Diabetic", f"{diabetic_count}/9")
    col_consensus2.metric("Models Predict Healthy", f"{healthy_count}/9")

    # ========= VISUAL COMPARISON =========
    st.markdown("---")
    st.subheader("Visual Comparisons")

    # Bar chart for diabetes probabilities
    from src.model_evaluation import plot_diabetes_probability_chart

    fig_prob = plot_diabetes_probability_chart(patient_predictions)
    st.plotly_chart(fig_prob, use_container_width=True)

    # Accuracy comparison
    comparison_df = create_comparison_table(eval_results, sort_by='accuracy')

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['Accuracy'],
        marker_color='lightblue',
        text=comparison_df['Accuracy'].apply(lambda x: f'{x:.2%}'),
        textposition='outside'
    ))

    fig_acc.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Algorithm',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        height=400
    )

    st.plotly_chart(fig_acc, use_container_width=True)

    # ========= CLINICAL INTERPRETATION =========
    st.markdown("---")
    st.subheader("Clinical Interpretation")

    with st.expander("View Detailed Analysis", expanded=True):
        st.markdown(f"""
        ### Patient Profile

        **Demographics:**
        - Age: {patient_data['Age']} years
        - Pregnancies: {patient_data['Pregnancies']}

        **Key Measurements:**
        - Glucose: {patient_data['Glucose']} mg/dL {"⚠️ (High)" if patient_data['Glucose'] > 140 else "✅ (Normal)"}
        - Blood Pressure: {patient_data['BloodPressure']} mm Hg {"⚠️ (High)" if patient_data['BloodPressure'] > 80 else "✅ (Normal)"}
        - BMI: {patient_data['BMI']:.1f} {"⚠️ (Obese)" if patient_data['BMI'] >= 30 else "⚠️ (Overweight)" if patient_data['BMI'] >= 25 else "✅ (Normal)"}
        - Insulin: {patient_data['Insulin']} mu U/ml
        - Skin Thickness: {patient_data['SkinThickness']} mm
        - Diabetes Pedigree: {patient_data['DiabetesPedigreeFunction']:.3f}

        ---

        ### Analysis Summary

        - **Best Model:** {best_model_name} (Accuracy: {best_model_metrics['accuracy']:.1%})
        - **Prediction:** {best_pred['prediction_label']}
        - **Confidence:** {best_pred['confidence']:.1f}% {f"(Diabetes Probability: {best_pred['diabetes_probability']:.1f}%)" if best_pred['confidence'] else ""}
        - **Model Consensus:** {diabetic_count}/9 models predict diabetic, {healthy_count}/9 predict healthy

        ---

        ### Risk Assessment

        {"⚠️ **High Risk Detected**" if best_pred['prediction'] == 1 else "✅ **Low Risk Detected**"}

        **Key Risk Factors Identified:**
        """)

        # Identify risk factors
        risk_factors = []
        if patient_data['Glucose'] > 140:
            risk_factors.append("- **High glucose level** (>140 mg/dL)")
        if patient_data['BMI'] >= 30:
            risk_factors.append("- **Obesity** (BMI ≥ 30)")
        elif patient_data['BMI'] >= 25:
            risk_factors.append("- **Overweight** (BMI ≥ 25)")
        if patient_data['BloodPressure'] > 90:
            risk_factors.append("- **High blood pressure** (>90 mm Hg)")
        if patient_data['Age'] > 45:
            risk_factors.append("- **Age over 45 years**")
        if patient_data['DiabetesPedigreeFunction'] > 0.5:
            risk_factors.append("- **High genetic predisposition**")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
                st.write("") # Add spacing
        else:
            st.markdown("- No major risk factors identified")

        st.markdown("""
        ---

        ### Recommendations
        """)

        if best_pred['prediction'] == 1:
            st.markdown("""
            - 🏥 **Consult a healthcare provider** for proper diabetes screening

            - 🩺 **Consider HbA1c test** and fasting glucose test

            - 🍎 **Implement dietary changes** (reduce sugar, increase fiber)

            - 🏃 **Increase physical activity** (150+ minutes/week)

            - ⚖️ **Monitor weight** and aim for healthy BMI

            - 📊 **Regular blood glucose monitoring**
            """)
        else:
            st.markdown("""
            - ✅ **Maintain healthy lifestyle** and regular checkups

            - 🍎 **Continue balanced diet**

            - 🏃 **Stay physically active**

            - ⚖️ **Maintain healthy weight**

            - 📅 **Annual health screenings recommended**
            """)

        st.markdown("""
        ---

        **⚠️ Important Disclaimer:** This is an educational machine learning tool and should NOT replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
        """)

    # ========= DOWNLOAD OPTIONS =========
    st.markdown("---")
    st.subheader("Download Results")

    col_down1, col_down2 = st.columns(2)

    with col_down1:
        # Download comparison table as CSV
        csv = pred_comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Table (CSV)",
            data=csv,
            file_name="diabetes_prediction_comparison.csv",
            mime="text/csv"
        )

    with col_down2:
        # Download patient data and prediction
        patient_results = pd.DataFrame([{
            **patient_data,
            'Best_Model': best_model_name,
            'Prediction': best_pred['prediction_label'],
            'Confidence': best_pred['confidence'],
            'Diabetes_Probability': best_pred['diabetes_probability']
        }])

        patient_csv = patient_results.to_csv(index=False)
        st.download_button(
            label="Download Patient Results (CSV)",
            data=patient_csv,
            file_name="patient_diabetes_prediction.csv",
            mime="text/csv"
        )
