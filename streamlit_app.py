# app.py - Simple Heart Disease Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter your medical information below to get a prediction")

# ============================================
# LOAD MODELS (from the same folder as app.py)
# ============================================
@st.cache_resource
def load_models():
    """Load all the saved model files"""
    try:
        # Get the current folder path
        current_folder = os.path.dirname(os.path.abspath(__file__))
        
        # Load model and preprocessing files
        model = joblib.load(os.path.join(current_folder, 'best_model_xgboost.pkl'))
        
        with open(os.path.join(current_folder, 'preprocess_patient.pkl'), 'rb') as f:
            preprocess_fn = pickle.load(f)
        
        return model, preprocess_fn
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure all .pkl files are in the same folder as app.py")
        return None, None

model, preprocess_fn = load_models()

# ============================================
# INPUT FORM (User fills this out)
# ============================================
st.subheader("Patient Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=50)
    sex = st.radio("Sex", ["Female", "Male"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-anginal Pain"]
    )
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120", ["No", "Yes"])

with col2:
    ekg = st.selectbox(
        "EKG Results",
        ["Normal", "ST-T Abnormality", "LVH"]
    )
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.radio("Exercise Induced Angina", ["No", "Yes"])
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    slope = st.selectbox(
        "Slope of ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thallium = st.selectbox(
        "Thallium Stress Test",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )

# ============================================
# Convert user inputs to model format
# ============================================
# Map text selections to numeric values (matching your training data)
chest_pain_map = {
    "Asymptomatic": 1,
    "Typical Angina": 2,
    "Atypical Angina": 3,
    "Non-anginal Pain": 4
}

ekg_map = {
    "Normal": 1,
    "ST-T Abnormality": 2,
    "LVH": 3
}

slope_map = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}

thallium_map = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

# ============================================
# PREDICT BUTTON
# ============================================
if st.button("Predict Heart Disease Risk", type="primary"):
    if model is None:
        st.error("Model not loaded. Please check your .pkl files.")
    else:
        # Prepare the raw data dictionary
        patient_data = {
            'Age': age,
            'Sex': 0 if sex == "Female" else 1,
            'Chest_pain_type': chest_pain_map[chest_pain],
            'BP': bp,
            'Cholesterol': cholesterol,
            'FBS_over_120': 0 if fbs == "No" else 1,
            'EKG_results': ekg_map[ekg],
            'Max_HR': max_hr,
            'Exercise_angina': 0 if exercise_angina == "No" else 1,
            'ST_depression': st_depression,
            'Slope_of_ST': slope_map[slope],
            'Number_of_vessels_fluro': vessels,
            'Thallium': thallium_map[thallium]
        }
        
        # Show what was entered
        with st.expander("View entered data"):
            st.json(patient_data)
        
        # Make prediction
        try:
            with st.spinner("Analyzing..."):
                processed = preprocess_fn(patient_data)
                probability = model.predict_proba(processed)[0][1]
                prediction = model.predict(processed)[0]
            
            # Show results
            st.subheader("Results")
            
            risk_percent = probability * 100
            
            # Display risk meter
            st.metric("Risk Probability", f"{risk_percent:.1f}%")
            st.progress(probability / 100)
            
            # Display final verdict
            if prediction == 1:
                st.error("⚠️ HIGH RISK - Heart Disease Detected")
                st.warning("Please consult a cardiologist for further evaluation.")
            else:
                st.success("✅ LOW RISK - No Heart Disease Detected")
                st.info("Maintain a healthy lifestyle with regular exercise and balanced diet.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================
# Sidebar Information
# ============================================
with st.sidebar:
    st.header("About")
    st.write("This model uses XGBoost to predict heart disease risk based on clinical data.")
    st.write("The preprocessing automatically creates engineered features like:")
    st.write("- Age/BP Ratio")
    st.write("- Cholesterol/HR Ratio")
    st.write("- Age Groups, BP Categories, etc.")
