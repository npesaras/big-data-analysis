"""
Real-Time Dynamic ML Training System for Diabetes Classification
Single-page interface with sidebar configuration, patient input, and live results
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import src.config as config
from src.pipeline import run_full_pipeline, display_results

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🏥 Diabetes Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

# =============================================================================
# 3. SIDEBAR: CONFIGURATION
# =============================================================================
with st.sidebar:
    st.title("🏥 Diabetes ML System")
    st.markdown("Real-time training & prediction")
    st.markdown("---")

    # ========= TRAINING CONFIGURATION =========
    st.subheader("⚙️ Training Configuration")

    # Train/Test Split
    train_size = st.slider(
        "Training Data Size (%)",
        min_value=60,
        max_value=90,
        value=80,
        step=5,
        help="Percentage of data used for training models"
    )
    test_size = 100 - train_size

    st.caption(f"📊 Train: {train_size}% | Test: {test_size}%")

    # K-Neighbors for KNN
    k_neighbors = st.number_input(
        "K-Neighbors (for KNN algorithm)",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of neighbors for K-Nearest Neighbors algorithm"
    )

    # Random Seed
    random_seed = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=100,
        value=42,
        help="For reproducible results"
    )

    st.markdown("---")

    # ========= PREPROCESSING OPTIONS =========
    st.subheader("⚙️ Preprocessing Options")

    imputation_strategy = st.selectbox(
        "Imputation Strategy",
        options=["median", "mean"],
        index=0,
        help="Strategy for handling missing values"
    )

    handle_zeros = st.checkbox(
        "Handle Zero Values",
        value=True,
        help="Replace biologically impossible zeros with NaN"
    )

    st.markdown("---")

    # ========= DATASET INFO =========
    st.subheader("📊 Dataset Info")

    if st.session_state.dataset_info:
        info = st.session_state.dataset_info
        st.metric("Total Samples", info['total'])
        st.caption(f"Training: {info['train']} ({info['train_pct']:.0f}%)")
        st.caption(f"Test: {info['test']} ({info['test_pct']:.0f}%)")
    else:
        st.info("Run analysis to see data split")

    st.markdown("---")

    # ========= INFO SECTION =========
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        ### Real-Time ML Pipeline

        1. **Configure** parameters above
        2. **Enter** patient data →
        3. **Click** "Predict & Analyze"
        4. **System executes:**
           - Load 768 patient dataset
           - Split train/test
           - Preprocess data
           - Train 9 algorithms
           - Evaluate & predict
        5. **View** comprehensive results

        ⏱️ **Time:** 10-30 seconds

        🔄 **Fresh Training:** Models are trained from scratch each time with your custom settings.
        """)

# =============================================================================
# 4. MAIN CONTENT AREA
# =============================================================================
st.title('🏥 Diabetes Classification System')
st.markdown('**Real-time ML training and prediction system**')
st.markdown("Configure parameters in the sidebar, enter patient data below, and click the button to train 9 algorithms and get predictions.")
st.markdown("---")

# Create two columns: Patient Input (left) and Results (right)
col_input, col_results = st.columns([2, 3])

# ========= LEFT COLUMN: PATIENT INPUT =========
with col_input:
    st.subheader('📋 Patient Information')
    st.markdown('Enter patient diagnostic measurements:')

    # Basic Information
    with st.container(border=True):
        st.markdown('#### Basic Information')
        pregnancies = st.number_input(
            'Pregnancies',
            min_value=0,
            max_value=17,
            value=3,
            help='Number of times pregnant'
        )
        age = st.number_input(
            'Age',
            min_value=21,
            max_value=81,
            value=33,
            help='Age in years'
        )

    # Clinical Measurements
    with st.container(border=True):
        st.markdown('#### Clinical Measurements')
        glucose = st.number_input(
            'Glucose',
            min_value=0,
            max_value=199,
            value=148,
            help='Plasma glucose concentration (mg/dL) - 0 indicates missing data'
        )
        if glucose < 70:
            st.warning('⚠️ Glucose below 70 mg/dL indicates hypoglycemia')
        elif glucose > 140:
            st.warning('⚠️ Glucose above 140 mg/dL may indicate hyperglycemia')

        blood_pressure = st.number_input(
            'Blood Pressure',
            min_value=0,
            max_value=122,
            value=72,
            help='Diastolic blood pressure (mm Hg) - 0 indicates missing data'
        )
        if blood_pressure < 60:
            st.warning('⚠️ Blood pressure below 60 mm Hg is unusually low')
        elif blood_pressure > 90:
            st.warning('⚠️ Blood pressure above 90 mm Hg may indicate hypertension')

        skin_thickness = st.number_input(
            'Skin Thickness',
            min_value=0,
            max_value=99,
            value=35,
            help='Triceps skin fold thickness (mm) - 0 indicates missing data'
        )

        insulin = st.number_input(
            'Insulin',
            min_value=0,
            max_value=846,
            value=0,
            help='2-Hour serum insulin (mu U/ml) - 0 indicates missing data'
        )

        bmi = st.number_input(
            'BMI',
            min_value=0.0,
            max_value=67.1,
            value=33.6,
            step=0.1,
            help='Body mass index (weight in kg/(height in m)^2) - 0 indicates missing data'
        )
        if bmi < 18.5:
            st.warning('⚠️ BMI below 18.5 indicates underweight')
        elif bmi >= 30.0:
            st.warning('⚠️ BMI above 30.0 indicates obesity')

        dpf = st.number_input(
            'Diabetes Pedigree Function',
            min_value=0.078,
            max_value=2.420,
            value=0.627,
            step=0.001,
            format='%.3f',
            help='Diabetes pedigree function (genetic influence)'
        )

    # Main Action Button
    st.markdown("---")
    predict_button = st.button(
        '🔍 Predict & Analyze',
        type='primary',
        use_container_width=True,
        help='Run full ML pipeline: Load → Clean → Train → Predict'
    )

# ========= RIGHT COLUMN: RESULTS =========
with col_results:
    st.subheader('🎯 Analysis Results')

    if predict_button:
        # Prepare patient data
        patient_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }

        # Execute full ML pipeline
        with st.spinner('🔄 Running complete ML pipeline...'):
            results = run_full_pipeline(
                patient_data=patient_data,
                train_size=train_size / 100,
                k_neighbors=k_neighbors,
                random_seed=random_seed,
                imputation_strategy=imputation_strategy,
                handle_zeros=handle_zeros
            )

        # Store dataset info in session state
        if results['success'] and results['dataset_info']:
            st.session_state.dataset_info = results['dataset_info']

        # Display results
        if results['success']:
            st.success('✅ Analysis complete!')
            display_results(results, patient_data, train_size / 100)
        else:
            st.error(f"❌ Pipeline failed: {results['error']}")
            st.info("💡 Try adjusting the configuration parameters in the sidebar.")

    else:
        # Show placeholder before button click
        st.info('👈 Configure parameters in the sidebar and enter patient data')

        st.markdown("""
        ### How it works:

        1. **Configure** training parameters in the sidebar:
           - Adjust train/test split ratio
           - Set K-neighbors for KNN algorithm
           - Choose imputation strategy
           - Enable/disable zero value handling

        2. **Enter** patient measurements in the left panel:
           - Demographics (age, pregnancies)
           - Clinical measurements (glucose, BP, BMI, etc.)

        3. **Click** "Predict & Analyze" button

        4. **Wait** for the full ML pipeline to execute:
           - Load diabetes dataset (768 patients)
           - Clean and preprocess data
           - Split into train/test sets based on your configuration
           - Train all 9 algorithms from scratch
           - Evaluate models on test set
           - Make predictions for your patient

        5. **View** comprehensive results:
           - Best performing model identification
           - Prediction for your patient with confidence scores
           - All 9 models comparison table
           - Visual charts and analysis
           - Clinical interpretation and recommendations

        ---

        ⏱️ **Expected processing time:** 10-30 seconds

        🔄 **Fresh Training:** Every click trains new models with your custom settings - no pre-trained models used!

        🎓 **Educational Tool:** Perfect for understanding how different ML algorithms perform on medical data.
        """)

        # Show example patient profile
        with st.expander("📖 Example Patient Profile"):
            st.markdown("""
            **Sample Values (Click "Predict & Analyze" to test):**

            | Feature | Value | Interpretation |
            |---------|-------|----------------|
            | Pregnancies | 3 | Multiple pregnancies |
            | Glucose | 148 mg/dL | ⚠️ High (normal < 140) |
            | Blood Pressure | 72 mm Hg | ✅ Normal |
            | Skin Thickness | 35 mm | Normal |
            | Insulin | 0 | Missing (will be imputed) |
            | BMI | 33.6 | ⚠️ Obese (normal < 30) |
            | Diabetes Pedigree | 0.627 | Moderate genetic risk |
            | Age | 33 years | Young adult |

            **Expected Risk:** High (due to glucose level and obesity)
            """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p><strong>ITD105 - Big Data Analytics | Diabetes Classification</strong></p>
    <p>Real-Time Dynamic ML Training System with 9 Algorithms</p>
    <p><em>PIMA Indians Diabetes Dataset</em></p>
</div>
""", unsafe_allow_html=True)
