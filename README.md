# bio-fusion hackathon winner - 2026

# Preeclampsia Risk Assessment System  
### Dual-Model, Clinically Guided Machine Learning Framework

This repository presents a **two-layer machine learning system** for preeclampsia risk assessment, combining **unsupervised phenotyping for clinicians** and **supervised early risk screening for patients**. The system is designed to align with medical knowledge, clinical workflows, and real-world deployment constraints.

---

## Project Overview

Preeclampsia is a complex, multi-system hypertensive disorder of pregnancy. Rather than relying on a single predictive model, this project adopts a **dual-dashboard approach**:

- **Model A (Patient Dashboard)**  
  Early screening using supervised learning with high sensitivity.
- **Model B (Doctor Dashboard)**  
  Phenotype-driven risk stratification using unsupervised clustering followed by supervised classification.

Together, these models support **early detection, clinical interpretation, and risk-based decision making**.

---

## System Architecture
Patient Inputs ──▶ Model A (Supervised NN) ──▶ Risk Score + Action Plan
│
▼
Clinician Inputs ─▶ Model B (Clustering + RF) ─▶ Phenotype-Driven Risk Stratification


---

## Dataset Description

### Core Columns

**Demographic & Anthropometric**
- `age`, `gest_age`, `height`, `weight`, `bmi`

**Vital Signs**
- `sysbp`, `diabp`

**Laboratory & Biomarkers**
- `hb`, `pcv`, `tsh`, `platelet`, `creatinine`
- `plgf:sflt`, `seng`, `cysc`, `pp_13`, `glycerides`

**Binary Clinical Risk Factors**
- `htn`, `diabetes`, `fam_htn`, `sp_art`

**Lifestyle & Behavioral**
- `occupation`, `diet`, `activity`, `sleep`

---

## Model A – BioFusion-NN (Patient Dashboard)

### Objective
Early **high-sensitivity screening** of preeclampsia risk using readily available clinical and contextual features.

### Key Characteristics
- Supervised **Neural Network (MLPClassifier)**
- Optimized for **recall (sensitivity)**
- Designed for **patient-facing dashboards**
- Includes **explainability (SHAP)**

### Feature Engineering
- Mean Arterial Pressure (MAP)
- Synthetic environmental factors:
  - Heat Exposure
  - Air Pollution
  - Access to Care
- Domain knowledge injection (e.g., heat sensitivity in hypertensive patients)

### Training Strategy
- 80/20 train-test split (stratified)
- 5-fold cross-validated grid search
- Recall-optimized hyperparameter tuning

### Evaluation Metrics
- ROC-AUC
- Precision-Recall Curve (preferred for medical screening)
- Classification Report

### Explainability
- Random Forest surrogate model
- SHAP summary plots for global feature importance

### Deployment Output
- Serialized model: `biofusion_model_v1.pkl`
- Risk categories:
  - 🟢 Low Risk – Routine Care
  - 🟡 Moderate Risk – Increased Monitoring
  - 🔴 High Risk – Immediate Intervention

---

##  Model B – PE-PhenoRisk (Doctor Dashboard)

### Objective
Discover **latent clinical phenotypes** of preeclampsia and enable **rapid phenotype-based risk prediction**.

### Step 1: Unsupervised Phenotyping
- **K-Means clustering (k=2)**
- Features:
  - `sysbp`, `diabp`
  - `plgfsflt`, `pp_13`
  - `creatinine`, `seng`, `cysc`
- Median imputation + z-score scaling
- Silhouette score used for cluster quality assessment

### Step 2: Risk Phenotype Assignment
- Cluster centers interpreted as representative patient profiles
- Weighted scoring of biomarkers
- Clusters labeled:
  - `0` → Low Risk Phenotype
  - `1` → High Risk Phenotype

### Step 3: Supervised Classification
- Random Forest classifier
- Inputs: All numeric + binary features
- Target: Phenotype group
- Class-balanced training

### Model Outputs
- Phenotype label
- Risk probability
- Feature importance (clinical interpretability)

### Deployment Artifacts
- `preeclampsia_phenotype_model.pkl`
- Preprocessing objects (scalers, imputers)
- Feature schema

---

## Model Evaluation Summary

| Model | Primary Goal | Key Metric | Performance |
|-----|-------------|-----------|-------------|
| Model A | Early Screening | Recall / PR-AUC | High |
| Model B | Clinical Stratification | ROC-AUC | ~0.96 |

---

##  Dashboards

### Patient Dashboard
- Simple input form
- Risk probability + action plan
- Powered by **Model A**

### Doctor Dashboard
- Advanced biomarker inputs
- Phenotype interpretation
- Feature importance & cluster insights
- Powered by **Model B**

---

##  Clinical Disclaimer

This project is intended for **research and educational purposes only**.  
It is **not a diagnostic tool** and should not replace professional medical judgment.

---

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```
###  Key Contributions

- Dual-model clinical ML architecture

- Phenotype-driven risk stratification

- High-recall patient screening

- Explainable AI for maternal health

- Deployment-ready pipeline

###  Author

Rashmi Paboda <br>
Computer Science & Engineering <br>
Maternal Health AI Research Project <br>

### If you find this project useful, consider starring the repository!
