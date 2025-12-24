# app_doctor.py -  Lab Confirmation Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(page_title="Doctor Lab Dashboard", layout="wide")
st.title("Preeclampsia Lab Confirmation System ")

@st.cache_resource
def load_modelA():
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    project_root = os.path.dirname(script_dir)                
    models_dir = os.path.join(project_root, "models")         

    model_path = os.path.join(models_dir, "preeclampsia_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")


    if not os.path.exists(model_path):
        st.error(f" Model not found: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f" Scaler not found: {scaler_path}")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Exact 21 training features from your Model A notebook [file:1]
    feature_order = [
        "age", "gest_age", "height", "weight", "bmi",
        "sysbp", "diabp", "hb", "pcv", "tsh", "platelet",
        "creatinine", "plgfsflt", "SEng", "cysC", "pp_13", "glycerides",
        "htn", "diabetes", "fam_htn", "sp_art",
    ]

    return model, scaler, feature_order


modelA, scalerA, feature_order = load_modelA()

# Doctor input UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical & Lab Inputs (Part 1)")

    age = st.number_input("Age (years)", min_value=15, max_value=50, value=28, step=1)
    gest_age = st.number_input("Gestational Age (weeks)", min_value=8, max_value=40, value=11, step=1)
    height = st.number_input("Height (cm)", min_value=130, max_value=190, value=155, step=1)

    weight = st.number_input("Weight (kg)", min_value=35.0, max_value=120.0, value=65.0, step=0.5)
    bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=27.0, step=0.1)

    sysbp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120, step=1)
    diabp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80, step=1)

    hb = st.number_input("Hemoglobin (g/dL)", min_value=6.0, max_value=18.0, value=12.5, step=0.1)
    pcv = st.number_input("PCV (%)", min_value=20.0, max_value=55.0, value=38.0, step=0.5)
    tsh = st.number_input("TSH (µIU/mL)", min_value=0.1, max_value=15.0, value=2.5, step=0.1)
    platelet = st.number_input("Platelets (×10³/µL)", min_value=50, max_value=600, value=250, step=10)

with col2:
    st.subheader("Clinical & Lab Inputs (Part 2)")

    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
    plgfsflt = st.number_input("PlGF/sFlt Ratio", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
    SEng = st.number_input("sEng (ng/mL)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    cysC = st.number_input("Cystatin C", min_value=0.2, max_value=5.0, value=0.8, step=0.1)
    pp_13 = st.number_input("PP_13", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
    glycerides = st.number_input("Glycerides", min_value=50.0, max_value=500.0, value=150.0, step=5.0)

    htn = st.selectbox("Hypertension History", [0, 1])
    diabetes = st.selectbox("Diabetes History", [0, 1])
    fam_htn = st.selectbox("Family History of HTN", [0, 1])
    sp_art = st.selectbox("Assisted Reproduction (1 = Yes)", [0, 1])

# Predict button
if st.button("Confirm Lab Risk", type="primary"):
    # Build dict using EXACT 21 features (same names/order as training) [file:1]
    row = {
        "age": age,
        "gest_age": gest_age,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "sysbp": sysbp,
        "diabp": diabp,
        "hb": hb,
        "pcv": pcv,
        "tsh": tsh,
        "platelet": platelet,
        "creatinine": creatinine,
        "plgfsflt": plgfsflt,
        "SEng": SEng,
        "cysC": cysC,
        "pp_13": pp_13,
        "glycerides": glycerides,
        "htn": int(htn),
        "diabetes": int(diabetes),
        "fam_htn": int(fam_htn),
        "sp_art": int(sp_art),
    }

    df_row = pd.DataFrame([row])[feature_order]
    X_input = df_row.values.astype(float)

    if hasattr(modelA, 'feature_names_in_'):
        expected = list(modelA.feature_names_in_)
    else:
        expected = feature_order

    # Warn if we need to fill missing expected features
    missing = [c for c in expected if c not in df_row.columns]
    if missing:
        st.warning(f"Filling missing features with zeros: {', '.join(missing)}")

    df_row = df_row.reindex(columns=expected, fill_value=0)

    X_scaled = scalerA.transform(df_row)
    prob = modelA.predict_proba(X_scaled)[0][1]

    colL, colR = st.columns(2)
    with colL:
        st.metric("Lab-Confirmed Risk", f"{prob:.1%}")
    with colR:
        st.metric("Action", "🔴 HIGH" if prob > 0.6 else "🟢 LOW")

    if prob > 0.6:
        st.error(
            
            "🚨 **CONFIRMED HIGH RISK**\n\n"
            "- Start magnesium sulfate protocol\n"
            "- Admit to high-dependency or ICU\n"
            "- Intensify maternal–fetal monitoring"
        )
    else:
        st.success(
            "**Low Risk**\n\n"
            "- Continue routine antenatal care\n"
            "- Weekly BP and symptom monitoring\n"
            "- Reassess if clinical status changes"
        )
