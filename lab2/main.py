"""
ITD105 - Big Data Analytics
Lab Exercise #2 - Streamlit Web Application
ML Model Deployment for Classification and Regression Tasks
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Big Data Analytics Lab 2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. MAIN TITLE
# =============================================================================
st.title('🔬 ML Model Deployment for Lab 2')
st.markdown('### ITD105 - Big Data Analytics: Resampling Techniques & Performance Metrics')
st.markdown('**Team Members:** Barnett, Gumora, Quilantang, Tadoy, Ventic')
st.markdown('---')

# =============================================================================
# 3. LOAD MODELS WITH CACHING
# =============================================================================
@st.cache_resource
def load_classification_model():
    """Load the diabetes classification model"""
    return joblib.load('classification_model.joblib')

@st.cache_resource
def load_regression_model():
    """Load the housing price regression model"""
    return joblib.load('regression_model.joblib')

# Load models
try:
    classification_model = load_classification_model()
    regression_model = load_regression_model()
    st.success('✅ Models loaded successfully!')
except Exception as e:
    st.error(f'❌ Error loading models: {e}')
    st.stop()

# =============================================================================
# 4. CREATE TABS
# =============================================================================
tab1, tab2 = st.tabs(["📊 Part 1: Classification", "📈 Part 2: Regression"])

# =============================================================================
# TAB 1: CLASSIFICATION - DIABETES PREDICTION
# =============================================================================
with tab1:
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
                max_value=20,
                value=1,
                help='Number of times pregnant'
            )
            age = st.number_input(
                'Age',
                min_value=18,
                max_value=100,
                value=25,
                help='Age in years'
            )
            
            st.markdown('#### Clinical Measurements')
            glucose = st.number_input(
                'Glucose',
                min_value=0,
                max_value=300,
                value=120,
                help='Plasma glucose concentration (mg/dL)'
            )
            blood_pressure = st.number_input(
                'Blood Pressure',
                min_value=0,
                max_value=200,
                value=70,
                help='Diastolic blood pressure (mm Hg)'
            )
            skin_thickness = st.number_input(
                'Skin Thickness',
                min_value=0,
                max_value=100,
                value=20,
                help='Triceps skin fold thickness (mm)'
            )
            insulin = st.number_input(
                'Insulin',
                min_value=0,
                max_value=900,
                value=80,
                help='2-Hour serum insulin (mu U/ml)'
            )
            bmi = st.number_input(
                'BMI',
                min_value=0.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
                help='Body mass index (weight in kg/(height in m)^2)'
            )
            dpf = st.number_input(
                'Diabetes Pedigree Function',
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help='Diabetes pedigree function (genetic influence)'
            )
    
    with col2:
        st.subheader('🎯 Prediction Results')
        
        # Create prediction button
        if st.button('🔍 Predict Diabetes', type='primary', use_container_width=True):
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
                st.dataframe(input_data, use_container_width=True)
            
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
# TAB 2: REGRESSION - HOUSING PRICE PREDICTION
# =============================================================================
with tab2:
    st.header('🏠 Boston Housing Price Prediction')
    st.markdown('Predict median home value based on environmental and housing characteristics.')
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader('📋 Housing Information')
        st.markdown('Enter housing characteristics:')
        
        with st.container():
            st.markdown('#### Crime & Zoning')
            crim = st.number_input(
                'CRIM - Crime Rate',
                min_value=0.0,
                max_value=100.0,
                value=0.5,
                step=0.01,
                help='Per capita crime rate by town'
            )
            zn = st.number_input(
                'ZN - Residential Land',
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                help='Proportion of residential land zoned for lots over 25,000 sq.ft.'
            )
            indus = st.number_input(
                'INDUS - Non-Retail Business',
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.1,
                help='Proportion of non-retail business acres per town'
            )
            chas = st.selectbox(
                'CHAS - Charles River',
                options=[0, 1],
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help='Whether tract bounds Charles River (1 = bounds river; 0 = otherwise)'
            )
            
            st.markdown('#### Environmental Factors')
            nox = st.number_input(
                'NOX - Nitric Oxides',
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help='Nitric oxides concentration (parts per 10 million)'
            )
            
            st.markdown('#### Property Features')
            rm = st.number_input(
                'RM - Average Rooms',
                min_value=3.0,
                max_value=10.0,
                value=6.0,
                step=0.1,
                help='Average number of rooms per dwelling'
            )
            age = st.number_input(
                'AGE - Property Age',
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help='Proportion of owner-occupied units built prior to 1940'
            )
            
            st.markdown('#### Location & Access')
            dis = st.number_input(
                'DIS - Distance to Employment',
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                step=0.1,
                help='Weighted distances to five Boston employment centres'
            )
            rad = st.number_input(
                'RAD - Highway Access',
                min_value=1,
                max_value=24,
                value=5,
                help='Index of accessibility to radial highways'
            )
            tax = st.number_input(
                'TAX - Property Tax',
                min_value=100,
                max_value=800,
                value=300,
                help='Full-value property-tax rate per $10,000'
            )
            ptratio = st.number_input(
                'PTRATIO - Pupil-Teacher Ratio',
                min_value=10.0,
                max_value=25.0,
                value=18.0,
                step=0.1,
                help='Pupil-teacher ratio by town'
            )
            
            st.markdown('#### Demographics')
            b = st.number_input(
                'B - Black Population Proportion',
                min_value=0.0,
                max_value=400.0,
                value=350.0,
                step=1.0,
                help='1000(Bk - 0.63)^2 where Bk is proportion of Black residents'
            )
            lstat = st.number_input(
                'LSTAT - Lower Status Population',
                min_value=0.0,
                max_value=40.0,
                value=10.0,
                step=0.1,
                help='% lower status of the population'
            )
    
    with col2:
        st.subheader('🎯 Prediction Results')
        
        # Create prediction button
        if st.button('🔍 Predict Housing Price', type='primary', use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'CRIM': [crim],
                'ZN': [zn],
                'INDUS': [indus],
                'CHAS': [chas],
                'NOX': [nox],
                'RM': [rm],
                'AGE': [age],
                'DIS': [dis],
                'RAD': [rad],
                'TAX': [tax],
                'PTRATIO': [ptratio],
                'B': [b],
                'LSTAT': [lstat]
            })
            
            # Make prediction
            prediction = regression_model.predict(input_data)
            predicted_price = prediction[0] * 1000  # Convert to actual dollars
            
            # Display results
            st.markdown('---')
            st.success('✅ **Prediction Complete**')
            
            # Show main prediction metric
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                st.metric(
                    label="Predicted Housing Price",
                    value=f"${predicted_price:,.2f}"
                )
            
            with col_b:
                st.metric(
                    label="Price per Room",
                    value=f"${predicted_price/rm:,.2f}"
                )
            
            with col_c:
                st.metric(
                    label="Value (in $1000s)",
                    value=f"${prediction[0]:.2f}k"
                )
            
            st.markdown('---')
            
            # Display input summary
            with st.expander('📊 View Input Summary'):
                st.dataframe(input_data.T, use_container_width=True)
            
            # Property analysis
            with st.expander('💡 Property Analysis'):
                st.markdown(f"""
                **Property Overview:**
                - **Crime Rate:** {crim:.2f} ({"High" if crim > 5 else "Low"})
                - **Average Rooms:** {rm:.1f} rooms
                - **Property Age:** {age:.0f}% built pre-1940
                - **Distance to Employment:** {dis:.1f} units
                - **Environmental Quality (NOX):** {nox:.3f} ppm
                
                **Predicted Median Home Value:** ${predicted_price:,.2f}
                
                **Key Factors Influencing Price:**
                - Number of rooms (RM): More rooms typically increase value
                - Crime rate (CRIM): Lower crime rates increase value
                - Environmental quality (NOX): Better air quality increases value
                - Property tax rate (TAX): Can affect property values
                - Lower status population (LSTAT): Higher percentages may decrease value
                
                **Note:** This prediction is based on historical Boston housing data and 
                is for educational purposes. Actual prices may vary based on current market conditions.
                """)
        else:
            st.info('👈 Enter housing characteristics and click "Predict Housing Price" to see results.')
            
            # Show example values
            with st.expander('📖 Feature Descriptions & Typical Ranges'):
                st.markdown("""
                | Feature | Description | Typical Range |
                |---------|-------------|---------------|
                | **CRIM** | Per capita crime rate | 0.00632 - 88.9762 |
                | **ZN** | Proportion residential land zoned | 0 - 100 |
                | **INDUS** | Proportion non-retail business acres | 0.46 - 27.74 |
                | **CHAS** | Charles River dummy variable | 0 or 1 |
                | **NOX** | Nitric oxides concentration | 0.385 - 0.871 ppm |
                | **RM** | Average number of rooms | 3.561 - 8.780 |
                | **AGE** | Proportion built pre-1940 | 2.9 - 100% |
                | **DIS** | Distance to employment centers | 1.1296 - 12.1265 |
                | **RAD** | Accessibility to highways | 1 - 24 |
                | **TAX** | Property-tax rate per $10,000 | 187 - 711 |
                | **PTRATIO** | Pupil-teacher ratio | 12.6 - 22.0 |
                | **B** | Proportion of Black residents | 0.32 - 396.90 |
                | **LSTAT** | % lower status population | 1.73 - 37.97 |
                """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p><strong>ITD105 - Big Data Analytics | Lab Exercise #2</strong></p>
    <p>Utilizing Resampling Techniques and Performance Metrics</p>
    <p><em>Classification: K-Fold Cross-Validation | Regression: Repeated Random Train-Test Splits</em></p>
</div>
""", unsafe_allow_html=True)

