# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# TITLE
# ============================================
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("""
This app predicts the risk of heart disease based on patient clinical data.
Enter the patient information below to get a prediction.
""")

# ============================================
# LOAD ALL MODELS AND PREPROCESSING TOOLS
# ============================================
@st.cache_resource
def load_all_artifacts():
    """Load all saved models and preprocessing tools"""
    try:
        # Load the trained model
        model = joblib.load('best_model_xgboost.pkl')
        
        # Load preprocessing artifacts
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        numerical_cols = joblib.load('numerical_columns.pkl')
        categorical_cols = joblib.load('categorical_columns.pkl')
        
        # Load the preprocessing function
        with open('preprocess_patient.pkl', 'rb') as f:
            preprocess_fn = pickle.load(f)
        
        return model, scaler, feature_names, numerical_cols, categorical_cols, preprocess_fn
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        st.info("""
        Please make sure these files are uploaded to GitHub:
        - best_model_xgboost.pkl
        - scaler.pkl
        - feature_names.pkl
        - numerical_columns.pkl
        - categorical_columns.pkl
        - preprocess_patient.pkl
        """)
        return None, None, None, None, None, None

# Load everything
model, scaler, feature_names, numerical_cols, categorical_cols, preprocess_fn = load_all_artifacts()

# ============================================
# INPUT FORM
# ============================================
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=55, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Asymptomatic",
            2: "2 - Typical Angina",
            3: "3 - Atypical Angina",
            4: "4 - Non-anginal Pain"
        }[x]
    )
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=5)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200, step=10)
    fbs_over_120 = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    ekg_results = st.selectbox(
        "EKG Results",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Normal",
            2: "2 - ST-T Abnormality",
            3: "3 - LVH"
        }[x]
    )
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150, step=5)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    slope_of_st = st.selectbox(
        "Slope of ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Upsloping",
            2: "2 - Flat",
            3: "3 - Downsloping"
        }[x]
    )
    num_vessels = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
    thallium = st.selectbox(
        "Thallium Stress Test",
        options=[3, 6, 7],
        format_func=lambda x: {
            3: "3 - Normal",
            6: "6 - Fixed Defect",
            7: "7 - Reversible Defect"
        }[x]
    )

# ============================================
# PREDICTION BUTTON
# ============================================
st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    if model is None:
        st.error("❌ Models not loaded. Please upload all required .pkl files to GitHub.")
    else:
        # Prepare raw data dictionary (matches your preprocessing function)
        patient_raw_data = {
            'Age': age,
            'Sex': sex,
            'Chest_pain_type': chest_pain_type,
            'BP': bp,
            'Cholesterol': cholesterol,
            'FBS_over_120': fbs_over_120,
            'EKG_results': ekg_results,
            'Max_HR': max_hr,
            'Exercise_angina': exercise_angina,
            'ST_depression': st_depression,
            'Slope_of_ST': slope_of_st,
            'Number_of_vessels_fluro': num_vessels,
            'Thallium': thallium
        }
        
        # Show input summary
        with st.expander("📊 View Input Data"):
            st.json(patient_raw_data)
        
        # Preprocess and predict
        try:
            with st.spinner("Analyzing patient data..."):
                # Your preprocessing function creates engineered features automatically!
                processed_df = preprocess_fn(patient_raw_data)
                probability = model.predict_proba(processed_df)[0][1]
                prediction = model.predict(processed_df)[0]
            
            # ============================================
            # DISPLAY RESULTS
            # ============================================
            st.subheader("📊 Prediction Results")
            
            # Risk percentage with progress bar
            risk_percentage = probability * 100
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric("Risk Probability", f"{risk_percentage:.1f}%")
                st.progress(probability)
            
            with col2:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK")
                    st.write("Heart Disease Detected")
                else:
                    st.success("✅ LOW RISK")
                    st.write("No Heart Disease Detected")
            
            with col3:
                if probability >= 0.7:
                    st.markdown("🔴 **RISK LEVEL: VERY HIGH**")
                elif probability >= 0.5:
                    st.markdown("🟠 **RISK LEVEL: HIGH**")
                elif probability >= 0.3:
                    st.markdown("🟡 **RISK LEVEL: MODERATE**")
                elif probability >= 0.1:
                    st.markdown("🟢 **RISK LEVEL: LOW**")
                else:
                    st.markdown("✅ **RISK LEVEL: VERY LOW**")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                **🏥 Immediate Recommendations:**
                - Consult a cardiologist as soon as possible
                - Schedule a comprehensive heart health checkup
                - Consider lifestyle modifications (diet, exercise, stress management)
                - Follow up with regular monitoring
                """)
            else:
                st.info("""
                **💪 Preventive Recommendations:**
                - Maintain a heart-healthy diet
                - Exercise regularly (30 minutes, 5 days/week)
                - Monitor blood pressure and cholesterol annually
                - Avoid smoking and limit alcohol consumption
                """)
            
            # Show what features were created by preprocessing
            with st.expander("🔧 Engineered Features Created"):
                st.write("Your preprocessing function automatically created these features:")
                st.write("- Age_BP_Ratio")
                st.write("- Chol_HR_Ratio")
                st.write("- Age_Group (Young/Middle/Senior/Elderly)")
                st.write("- BP_Category (Normal/Elevated/High1/High2)")
                st.write("- Chol_Risk (Normal/Borderline/High)")
                st.write("- HR_Category (Low/Moderate/Good/Excellent)")
                
                # Show actual values for this patient
                st.write("**Actual values for this patient:**")
                for col in ['Age_BP_Ratio', 'Chol_HR_Ratio']:
                    if col in processed_df.columns:
                        st.write(f"- {col}: {processed_df[col].values[0]:.4f}")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Check that all model files are properly formatted and match the preprocessing function.")

# ============================================
# SIDEBAR WITH INFO
# ============================================
with st.sidebar:
    st.header("ℹ️ About This Model")
    st.markdown("""
    **Heart Disease Risk Predictor**
    
    This model uses **XGBoost** to predict heart disease risk with advanced feature engineering.
    
    **Engineered Features:**
    - Age/BP Ratio
    - Cholesterol/HR Ratio
    - Age Groups (Young, Middle, Senior, Elderly)
    - BP Categories
    - Cholesterol Risk Levels
    - Heart Rate Categories
    
    **Input ranges:**
    - Age: 18-120 years
    - BP: 80-200 mm Hg
    - Cholesterol: 100-400 mg/dl
    - Max HR: 60-220 bpm
    
    **Model files needed:**
    - best_model_xgboost.pkl
    - scaler.pkl
    - feature_names.pkl
    - numerical_columns.pkl
    - categorical_columns.pkl
    - preprocess_patient.pkl
    """)
    
    st.header("📋 Feature Legend")
    st.markdown("""
    **Chest Pain Types:**
    - 1: Asymptomatic (no symptoms)
    - 2: Typical Angina
    - 3: Atypical Angina
    - 4: Non-anginal Pain
    
    **Thallium Results:**
    - 3: Normal
    - 6: Fixed Defect
    - 7: Reversible Defect
    
    **EKG Results:**
    - 1: Normal
    - 2: ST-T Abnormality
    - 3: LVH
    """)
