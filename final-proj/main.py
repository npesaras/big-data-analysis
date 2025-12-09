"""
Real-Time Dynamic ML Training System for Diabetes Classification
Final Version: Sidebar Inputs + Fixed Icon Warnings
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
# 1. PAGE CONFIGURATION & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Diabetes Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Look
st.markdown("""
<style>
    /* Main container padding adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Header Styling */
    h1, h2, h3 {
        font-weight: 600;
    }

    /* Card Style for Results */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
    }

    /* Primary Button Styling */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    /* Sidebar specific adjustments */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Warning Box Styling */
    .stAlert {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

# =============================================================================
# 3. SIDEBAR: CONFIGURATION & INPUTS
# =============================================================================
with st.sidebar:
    st.title("🏥 Diabetes ML")
    st.caption("Configuration & Input")
    st.markdown("---")

    # ========= A. SYSTEM SETTINGS =========
    st.subheader("⚙️ System Settings")

    with st.expander("Model Configuration", expanded=False):
        train_size = st.slider("Train Size (%)", 60, 90, 80, 5)
        k_neighbors = st.number_input("KNN Neighbors", 1, 20, 5)
        random_seed = st.number_input("Random Seed", 1, 100, 42)
        imputation_strategy = st.selectbox("Imputation", ["median", "mean"])
        handle_zeros = st.checkbox("Handle Zeros", value=True)

    st.markdown("---")

    # ========= B. PATIENT DIAGNOSTICS =========
    st.subheader("📋 Patient Diagnostics")

    with st.form("patient_data_form"):
        st.caption("Enter patient metrics below:")

        # Group 1: Demographics
        st.markdown("**1. Demographics**")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input('Age', 21, 81, 33)
        with c2:
            pregnancies = st.number_input('Preg.', 0, 17, 3)

        # Group 2: Vitals
        st.markdown("**2. Vitals Signs**")
        c3, c4 = st.columns(2)
        with c3:
            glucose = st.number_input('Glucose', 0, 199, 148, help="mg/dL")
        with c4:
            blood_pressure = st.number_input('BP', 0, 122, 72, help="mm Hg")

        # Group 3: Body Metrics
        st.markdown("**3. Body Metrics**")
        c5, c6 = st.columns(2)
        with c5:
            bmi = st.number_input('BMI', 0.0, 67.1, 33.6, 0.1)
        with c6:
            skin_thickness = st.number_input('Skin', 0, 99, 35, help="mm")

        # Group 4: Advanced
        st.markdown("**4. Advanced**")
        c7, c8 = st.columns(2)
        with c7:
            insulin = st.number_input('Insulin', 0, 846, 0, help="mu U/ml")
        with c8:
            dpf = st.number_input('Pedigree', 0.078, 2.420, 0.627, 0.001, format='%.3f')

        st.markdown("---")

        # --- FIXED WARNINGS (Single Icon) ---
        if glucose > 140:
            st.warning('High Glucose Detected', icon="⚠️")
        if bmi >= 30.0:
            st.warning('Obesity Risk Indicated', icon="⚠️")
        # ------------------------------------

        # Submit Button
        predict_button = st.form_submit_button(
            '🔍 Run Analysis',
            type='primary',
            use_container_width=True
        )

    # ========= C. DATASET STATS =========
    if st.session_state.dataset_info:
        st.markdown("---")
        st.caption("📊 Current Data Split")
        info = st.session_state.dataset_info
        dc1, dc2 = st.columns(2)
        dc1.metric("Train", info['train'])
        dc2.metric("Test", info['test'])

# =============================================================================
# 4. MAIN CONTENT AREA (FULL WIDTH)
# =============================================================================

# Header Section
st.title('🏥 Diabetes Classification System')
st.markdown("### Real-Time Dynamic ML Training & Prediction")

# Logic to handle view
if predict_button:
    # ---------------------------------------------------------
    # VIEW: RESULTS (Full Width)
    # ---------------------------------------------------------
    st.divider()

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
    with st.spinner('🔄 Processing: Training 9 Algorithms & Analyzing Patient Data...'):
        results = run_full_pipeline(
            patient_data=patient_data,
            train_size=train_size / 100,
            k_neighbors=k_neighbors,
            random_seed=random_seed,
            imputation_strategy=imputation_strategy,
            handle_zeros=handle_zeros
        )

    # Store dataset info
    if results['success'] and results['dataset_info']:
        st.session_state.dataset_info = results['dataset_info']

    # Display results
    if results['success']:
        with st.container(border=True):
            st.subheader("🎯 Analysis Report")
            display_results(results, patient_data, train_size / 100)
    else:
        st.error(f"❌ Pipeline failed: {results['error']}")

else:
    # ---------------------------------------------------------
    # VIEW: LANDING / INSTRUCTION (Full Width)
    # ---------------------------------------------------------
    st.divider()

    # Hero Section Message
    st.info('👈 **Action Required:** Please enter patient diagnostics in the sidebar and click "Run Analysis".')

    # Visual Guide
    with st.container(border=True):
        st.markdown("### 🧬 System Architecture")

        col_step1, col_step2, col_step3 = st.columns(3)

        with col_step1:
            st.markdown("#### 1. Input & Configure")
            st.write("User defines patient vitals and adjusts ML hyperparameters (split ratio, KNN neighbors) in the sidebar.")

        with col_step2:
            st.markdown("#### 2. Dynamic Training")
            st.write("The system loads the PIMA dataset, cleans it, and **retrains 9 distinct algorithms** from scratch in real-time.")

        with col_step3:
            st.markdown("#### 3. Clinical Output")
            st.write("Generates risk probability, identifies the best performing model, and provides visual comparisons.")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>ITD105 - Big Data Analytics | Real-Time Medical ML System</p>
        </div>
        """,
        unsafe_allow_html=True
    )