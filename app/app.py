import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Clinical Dashboard", layout="wide")



@st.cache_resource
def load_model():
    here = os.path.abspath(os.path.dirname(__file__))  # app/
    project_root = os.path.abspath(os.path.join(here, '..'))
    cwd = os.path.abspath(os.getcwd())
    candidate_roots = [here, project_root, cwd, os.path.join(project_root, 'models'), os.path.join(here, 'models')]

    candidates = []
    for r in candidate_roots:
        candidates.append(os.path.join(r, 'biofusion_model_v1.pkl'))
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            unique_candidates.append(p)
            seen.add(p)

    for path in unique_candidates:
        if os.path.exists(path):
            return joblib.load(path), path

    raise FileNotFoundError("Model file 'biofusion_model_v1.pkl' not found. Checked paths:\n" + "\n".join(unique_candidates))

try:
    model, model_path = load_model()
    st.sidebar.info(f"Loaded model from: {model_path}")
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Place 'biofusion_model_v1.pkl' in the project root or a 'models/' directory, or update the path in this file.")
    st.stop()


st.sidebar.header("Patient Vitals Input")

def user_input_features():
    # --- Standard Clinical Vitals ---
    age = st.sidebar.slider("Age", 10, 70, 30)
    systolic_bp = st.sidebar.slider("Systolic BP (mmHg)", 70, 200, 120)
    diastolic_bp = st.sidebar.slider("Diastolic BP (mmHg)", 40, 120, 80)
    bs = st.sidebar.slider("Blood Sugar (mmol/L)", 6.0, 19.0, 7.0)
    body_temp = st.sidebar.slider("Body Temperature (°F)", 96.0, 105.0, 98.6)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 150, 75)
    
    # --- Medical History ---
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Medical History")
    
    bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 20.0)
    
    # Binary Questions (Yes/No)
    gest_diabetes = st.sidebar.selectbox("Gestational Diabetes?", ["No", "Yes"])
    pre_diabetes = st.sidebar.selectbox("Pre-existing Diabetes?", ["No", "Yes"])
    mental_health = st.sidebar.selectbox("History of Mental Health Issues?", ["No", "Yes"])
    prev_comp = st.sidebar.selectbox("History of Previous Complications?", ["No", "Yes"])
    
    # --- Environmental Factors ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Environmental Context")
    heat_exp = st.sidebar.selectbox("Heat Exposure Level", ["Low", "Medium", "High"])
    pollution = st.sidebar.selectbox("Air Quality Index", ["Low", "Medium", "High"])
    access = st.sidebar.selectbox("Access to Healthcare", ["Good", "Moderate", "Poor"])
    
    # MAPPINGS
    binary_map = {"No": 0, "Yes": 1}
    env_map = {"Low": 0, "Medium": 1, "High": 2, "Good": 2, "Moderate": 1, "Poor": 0}
    
    # Feature Engineering (MAP)
    map_val = (systolic_bp + 2 * diastolic_bp) / 3

    # CREATE RAW DATAFRAME
    # We create a dictionary with all the data we have
    data = {
        'Age': age,
        'Systolic BP': systolic_bp,
        'Diastolic': diastolic_bp,  
        'BS': bs,
        'Body Temp': body_temp,
        'Heart Rate': heart_rate,    
        'BMI': bmi,                  
        'Gestational Diabetes': binary_map[gest_diabetes], 
        'Preexisting Diabetes': binary_map[pre_diabetes],  
        'Mental Health': binary_map[mental_health],   
        'Previous Complications': binary_map[prev_comp],
        'MAP': map_val,
        'Heat_Exposure': env_map[heat_exp],
        'Air_Pollution': env_map[pollution],
        'Access_To_Care': env_map[access]
    }
    
    return pd.DataFrame(data, index=[0]), map_val

input_df, map_val = user_input_features()

# --- 3. MAIN DASHBOARD ---
st.title("Preeclampsia Early Warning System")
st.markdown("### Real-time Risk Stratification for Low-Resource Settings")

col1, col2 = st.columns([2, 1])

with col1:
    st.info(f"**Calculated Mean Arterial Pressure (MAP):** {map_val:.1f} mmHg")
    
    if st.button(' Assess Patient Risk'):
        try:
            
            # 1. Get the list of columns the model expects
            if hasattr(model, 'feature_names_in_'):
                expected_cols = model.feature_names_in_
            else:
                # Fallback for older scikit-learn versions: try to get from the first step of pipeline
                expected_cols = model.steps[0][1].feature_names_in_
            
            # 2. Reorder the dataframe to match the model (fill missing with 0)
            final_df = input_df.reindex(columns=expected_cols, fill_value=0)

            # Get Probability
            probability = model.predict_proba(final_df)[0][1]
            
            st.divider()
            
            # Dynamic Risk Assessment
            if probability >= 0.7:
                st.error(f"🔴 **HIGH RISK DETECTED** (Probability: {probability:.1%})")
                st.write("**Recommendation:** Immediate Clinical Referral & Magnesium Sulfate Protocol.")
            elif probability >= 0.3:
                st.warning(f"🟡 **Moderate Risk** (Probability: {probability:.1%})")
                st.write("**Recommendation:** Increase monitoring frequency to weekly.")
            else:
                st.success(f"🟢 **Low Risk** (Probability: {probability:.1%})")
                st.write("**Recommendation:** Standard Antenatal Care.")
                
        except Exception as e:
            st.error("⚠️ An error occurred.")
            st.write(f"**Error Details:** {e}")
            # Debugging helper: Show what the model wanted vs what it got
            if hasattr(model, 'feature_names_in_'):
                 st.write("**Expected Columns:**", list(model.feature_names_in_))

with col2:
    st.subheader(" Why this prediction?")
    st.write("Visualizing key vitals against population averages.")
    
    # Simple Bar Chart for the demo
    # Re-using input_df for visualization since it has friendly names
    chart_data = input_df[['Systolic BP', 'MAP', 'BS']].T
    chart_data.columns = ["Patient Value"]
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("*Powered by BioFusion Neural Network (v1.0)*")