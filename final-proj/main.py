import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import src.config as config
from src.utils import load_model

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🏥 Diabetes Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. MAIN TITLE
# =============================================================================
st.title('🏥 Diabetes Classification System')

# =============================================================================
# 3. LOAD MODELS WITH CACHING
# =============================================================================
@st.cache_resource
def load_classification_model():
    """Load the diabetes classification model"""
    model_path = config.MODELS_DIR / "classification_model.joblib"
    if model_path.exists():
        return load_model(model_path)
    return None

# Load model
try:
    classification_model = load_classification_model()
    if classification_model is not None:
        st.success('✅ Model loaded successfully!')
    else:
        st.warning('⚠️ No trained model found. Please train a model first using the training script.')
        st.info('💡 You can still use the interface to see how it works with sample predictions.')
except Exception as e:
    st.error(f'❌ Error loading model: {e}')
    classification_model = None

# =============================================================================
# 4. DIABETES PREDICTION INTERFACE
# =============================================================================
st.header('🏥 Pima Indians Diabetes Prediction')
st.markdown('Predict diabetes onset based on diagnostic measurements.')

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('📋 Patient Information')
    st.markdown('Enter patient diagnostic measurements:')

    # Sidebar for Classification inputs
    with st.container():
        st.markdown('#### Basic Information')
        pregnancies = st.number_input(
            'Pregnancies',
            min_value=0,
            max_value=17,
            value=1,
            help='Number of times pregnant (0-17)'
        )
        age = st.number_input(
            'Age',
            min_value=21,
            max_value=81,
            value=25,
            help='Age in years (21-81)'
        )

        st.markdown('#### Clinical Measurements')
        glucose = st.number_input(
            'Glucose',
            min_value=44,
            max_value=199,
            value=120,
            help='Plasma glucose concentration (mg/dL) [44-199]'
        )
        if glucose < 70:
            st.warning('⚠️ Glucose below 70 mg/dL indicates hypoglycemia')
        elif glucose > 140:
            st.warning('⚠️ Glucose above 140 mg/dL may indicate hyperglycemia')
        blood_pressure = st.number_input(
            'Blood Pressure',
            min_value=40,
            max_value=122,
            value=70,
            help='Diastolic blood pressure (mm Hg) [40-122]'
        )
        if blood_pressure < 60:
            st.warning('⚠️ Blood pressure below 60 mm Hg is unusually low')
        elif blood_pressure > 90:
            st.warning('⚠️ Blood pressure above 90 mm Hg may indicate hypertension')
        skin_thickness = st.number_input(
            'Skin Thickness',
            min_value=7,
            max_value=99,
            value=20,
            help='Triceps skin fold thickness (mm) [7-99]'
        )
        insulin = st.number_input(
            'Insulin',
            min_value=14,
            max_value=846,
            value=80,
            help='2-Hour serum insulin (mu U/ml) [14-846]'
        )
        bmi = st.number_input(
            'BMI',
            min_value=15.0,
            max_value=67.1,
            value=25.0,
            step=0.1,
            help='Body mass index (weight in kg/(height in m)^2) [15.0-67.1]'
        )
        if bmi < 18.5:
            st.warning('⚠️ BMI below 18.5 indicates underweight')
        elif bmi >= 30.0:
            st.warning('⚠️ BMI above 30.0 indicates obesity')
        dpf = st.number_input(
            'Diabetes Pedigree Function',
            min_value=0.078,
            max_value=2.420,
            value=0.500,
            step=0.001,
            format='%.3f',
            help='Diabetes pedigree function (genetic influence) [0.078-2.420]'
        )

with col2:
    st.subheader('🎯 Prediction Results')

    # Create prediction button
    if st.button('🔍 Predict Diabetes', type='primary', use_container_width=True):
        if classification_model is None:
            st.error('❌ No model available. Please train a model first.')
            st.stop()

        # Prepare input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Make prediction
        prediction = classification_model.predict(input_data)
        prediction_proba = classification_model.predict_proba(input_data)

        # Display results
        st.markdown('---')

        if prediction[0] == 1:
            st.error('⚠️ **Prediction: Diabetic**')
            confidence = prediction_proba[0][1] * 100
        else:
            st.success('✅ **Prediction: Not Diabetic**')
            confidence = prediction_proba[0][0] * 100

        # Show confidence metrics
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

        with col_b:
            st.metric(
                label="Diabetes Probability",
                value=f"{prediction_proba[0][1]*100:.2f}%"
            )

        with col_c:
            st.metric(
                label="Healthy Probability",
                value=f"{prediction_proba[0][0]*100:.2f}%"
            )

        st.markdown('---')

        # Display input summary
        with st.expander('📊 View Input Summary'):
            st.dataframe(input_data, width='stretch')

        # Clinical interpretation
        with st.expander('💡 Clinical Interpretation'):
            st.markdown(f"""
            **Patient Profile:**
            - **Age:** {age} years
            - **Pregnancies:** {pregnancies}
            - **Glucose Level:** {glucose} mg/dL {"(High)" if glucose > 140 else "(Normal)"}
            - **Blood Pressure:** {blood_pressure} mm Hg {"(High)" if blood_pressure > 80 else "(Normal)"}
            - **BMI:** {bmi:.1f} {"(Overweight)" if bmi > 25 else "(Normal)"}

            **Model Confidence:** {confidence:.2f}%

            **Note:** This prediction is for educational purposes only and should not replace
            professional medical advice. Please consult with a healthcare provider for proper diagnosis.
            """)
    else:
        st.info('👈 Enter patient data and click "Predict Diabetes" to see results.')

        # Show example values
        with st.expander('📖 Feature Descriptions & Normal Ranges'):
            st.markdown("""
            | Feature | Description | Normal Range |
            |---------|-------------|--------------|
            | **Pregnancies** | Number of times pregnant | 0-17 |
            | **Glucose** | Plasma glucose concentration | 70-140 mg/dL |
            | **Blood Pressure** | Diastolic blood pressure | 60-80 mm Hg |
            | **Skin Thickness** | Triceps skin fold thickness | 10-50 mm |
            | **Insulin** | 2-Hour serum insulin | 16-166 mu U/ml |
            | **BMI** | Body mass index | 18.5-24.9 |
            | **Diabetes Pedigree** | Genetic influence score | 0.0-2.5 |
            | **Age** | Age in years | 21-81 |
            """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p><strong>ITD105 - Big Data Analytics | Diabetes Classification</strong></p>
    <p>Utilizing K-Fold and Leave-One-Out Cross-Validation</p>
    <p><em>PIMA Indians Diabetes Dataset Classification Model</em></p>
</div>
""", unsafe_allow_html=True)

