import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score

# Set page config for a wider layout
st.set_page_config(page_title="CVD Risk Prediction", layout="wide")

# Function to download a file if it doesn’t exist
def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

# URLs for model files
stacking_model_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/stacking_genai_model.pkl'
scaler_url = 'https://raw.githubusercontent.com/HowardHNguyen/genai/main/scaler.pkl'

# Local paths
stacking_model_path = 'stacking_genai_model.pkl'
scaler_path = 'scaler.pkl'

# Download models and scaler if not present
if not os.path.exists(stacking_model_path):
    st.info(f"Downloading {stacking_model_path}...")
    download_file(stacking_model_url, stacking_model_path)

if not os.path.exists(scaler_path):
    st.info(f"Downloading {scaler_path}...")
    download_file(scaler_url, scaler_path)

# Load the stacking model
@st.cache_resource
def load_stacking_model():
    try:
        loaded_object = joblib.load(stacking_model_path)
        if isinstance(loaded_object, dict) and 'gen_stacking_meta_model' in loaded_object:
            return {
                'meta_model': loaded_object['gen_stacking_meta_model'],
                'base_models': {
                    'rf': loaded_object.get('rf_model'),
                    'xgb': loaded_object.get('xgb_model')
                }
            }
        else:
            st.error("Model structure incorrect. Please check model file.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

stacking_model = load_stacking_model()

# Load the scaler
scaler = joblib.load(scaler_path)

# Define feature columns
feature_columns = [
    'SEX', 'AGE', 'educ', 'CURSMOKE', 'CIGPDAY', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRTE',
    'GLUCOSE', 'HDLC', 'LDLC', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'
]

# Title
st.title("🫀 Cardiovascular Disease (CVD) Risk Prediction")
st.write("This tool helps assess your potential risk of developing CVD based on clinical parameters.")

# Sidebar Inputs
st.sidebar.header("📋 Input Your Health Metrics")

user_data = {
    'SEX': st.sidebar.selectbox("SEX (0 = Female, 1 = Male)", [0, 1], index=1),
    'AGE': st.sidebar.slider("AGE", 32.0, 81.0, 35.0),
    'educ': st.sidebar.slider("Education Level (educ)", 1.0, 4.0, 3.0),
    'CURSMOKE': st.sidebar.selectbox("Current Smoker (0 = No, 1 = Yes)", [0, 1], index=0),
    'CIGPDAY': st.sidebar.slider("Cigarettes per Day", 0.0, 90.0, 0.0),
    'TOTCHOL': st.sidebar.slider("Total Cholesterol", 107.0, 696.0, 195.0),
    'SYSBP': st.sidebar.slider("Systolic BP", 83.5, 295.0, 120.0),
    'DIABP': st.sidebar.slider("Diastolic BP", 30.0, 150.0, 80.0),
    'BMI': st.sidebar.slider("BMI", 15.0, 56.8, 24.8),
    'HEARTRTE': st.sidebar.slider("Heart Rate", 37.0, 220.0, 60.0),
    'GLUCOSE': st.sidebar.slider("Glucose", 39.0, 478.0, 90.0),
    'HDLC': st.sidebar.slider("HDL Cholesterol", 10.0, 189.0, 60.0),
    'LDLC': st.sidebar.slider("LDL Cholesterol", 20.0, 565.0, 98.0),
    'DIABETES': st.sidebar.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1], index=0),
    'BPMEDS': st.sidebar.selectbox("BP Meds (0 = No, 1 = Yes)", [0, 1], index=0),
    'PREVCHD': st.sidebar.selectbox("Prev CHD (0 = No, 1 = Yes)", [0, 1], index=0),
    'PREVAP': st.sidebar.selectbox("PREVAP (0 = No, 1 = Yes)", [0, 1], index=0),
    'PREVMI': st.sidebar.selectbox("PREVMI (0 = No, 1 = Yes)", [0, 1], index=0),
    'PREVSTRK': st.sidebar.selectbox("PREVSTRK (0 = No, 1 = Yes)", [0, 1], index=0),
    'PREVHYP': st.sidebar.selectbox("PREVHYP (0 = No, 1 = Yes)", [0, 1], index=0)
}

input_df = pd.DataFrame([user_data])

# Scale input data
input_df_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Risk"):
    if stacking_model:
        try:
            rf_proba = stacking_model['base_models']['rf'].predict_proba(input_df_scaled)[:, 1]
            xgb_proba = stacking_model['base_models']['xgb'].predict_proba(input_df_scaled)[:, 1]
            meta_input = np.column_stack([rf_proba, xgb_proba])
            meta_proba = stacking_model['meta_model'].predict_proba(meta_input)[:, 1][0]

            # Risk Level Classification
            risk_level = "🟢 Low Risk" if meta_proba < 0.3 else "🟡 Moderate Risk" if meta_proba < 0.7 else "🔴 High Risk"
            st.metric(label="**CVD Risk Probability**", value=f"{meta_proba:.2%}")
            st.success(f"**Risk Level: {risk_level}**")

            # Feature Importance (RF)
            st.subheader("📊 CVD Risk Factors (Top 10)")
            rf_model = stacking_model['base_models']['rf']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-10:]
            plt.figure(figsize=(8, 3))
            plt.barh(np.array(feature_columns)[indices], importances[indices], color='gray')
            plt.xlabel("Importance")
            plt.title("Top 10 Important Features (RF)")
            st.pyplot(plt)

            # Model Performance (ROC Curve)
            st.subheader("📉 Model Performance")
            st.write("The model has been evaluated on a test dataset with an AUC of 0.96.")

            st.subheader("📌 Important Notes")
            st.info("""
                - This is an AI-based prediction tool, not a medical diagnosis.
                - Always consult your doctor for medical advice.
                - For **high-risk results**, it is strongly recommended to **consult with a physician**.
                - A healthy lifestyle including **diet, exercise, and regular medical checkups** can reduce cardiovascular risks.
            """)

            # Data Information Notes 
            st.subheader("📌 Predictive Notes")
            st.write("""
                     Predictive models aim to forecast the likelihood or timing of outcomes (e.g., cardiovascular disease, stroke) based on baseline data. The Framingham study is renowned for cardiovascular risk assessment, so predictors should be relevant to such outcomes. All baseline characteristics are potential predictors because they provide information about risk factors:
                     - 'SEX': Gender differences affect disease risk.
                     - 'AGE': Older age increases risk for many conditions.
                     - 'TOTCHOL', 'HDLC', 'LDLC': Cholesterol levels are key for heart disease prediction. 
                     - 'SYSBP', 'DIABP': Blood pressure is a major cardiovascular risk factor.
                     - 'CURSMOKE', 'CIGPDAY': Smoking is a strong predictor of cardiovascular and other diseases. 
                     - 'BMI': Obesity is linked to multiple health risks.
                     - 'DIABETES': A significant risk factor for cardiovascular events.
                     - 'BPMEDS': Indicates treated hypertension, affecting blood pressure interpretation.
                     - 'HEARTRTE': Resting heart rate reflects fitness and health.
                     - 'GLUCOSE': Elevated levels indicate metabolic issues.
                     - 'educ': Socioeconomic status influences health outcomes.
                     - 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP': Prior events strongly predict future events.
            """, unsafe_allow_html=True)

            # Data Information Notes 
            st.subheader("📌 Preventive Notes")
            st.write("""
                     Preventive attributes are modifiable risk factors. While all predictors contribute to risk assessment, the following are directly modifiable or indicate conditions amenable to intervention:
                     - 'CURSMOKE', 'CIGPDAY': Smoking cessation reduces risk.
                     - 'SYSBP', 'DIABP', 'BPMEDS': Blood pressure can be managed with lifestyle changes or medication.
                     - 'TOTCHOL', 'HDLC', 'LDLC': Cholesterol levels can be altered via diet, exercise, or drugs.
                     - 'BMI': Weight loss improves health outcomes.
                     - 'GLUCOSE', 'DIABETES': Glucose control prevents diabetes progression.
                     - 'HEARTRTE': Exercise can improve resting heart rate.
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.error("⚠️ Model loading failed. Please check the model file.")

st.write("Developed by **Howard Nguyen** | Data Science & AI | 2025")
