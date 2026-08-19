# Preeclampsia Risk Assessment System

stage 01 - https://twostagepreeclampsiaaimodelgit-djjfq29rhtv9h3khwqkcqw.streamlit.app/
stage 02 - https://twostagepreeclampsiaaimodelgit-76r4nus9gj7mxxvfxapput.streamlit.app/

**BioFusion Hackathon Winner — 2026**

A dual-model, clinically guided machine learning framework for preeclampsia screening and phenotype-based risk stratification.

Preeclampsia is a multi-system hypertensive disorder of pregnancy. This project does not treat it as a single prediction problem. Instead it ships two complementary models:

- **Model A (BioFusion-NN)** — high-sensitivity early screening from everyday vitals, for a patient-facing dashboard.
- **Model B (PE-PhenoRisk)** — unsupervised phenotype discovery plus supervised classification from lab and biomarker data, for a clinician dashboard.

Together they support early detection, clinical interpretation, and risk-based follow-up. This repository is for **research and education only**. It is not a diagnostic device and must not replace professional medical judgment.

---

## Table of contents

- [Why two models](#why-two-models)
- [System architecture](#system-architecture)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Model A — BioFusion-NN (patient dashboard)](#model-a--biofusion-nn-patient-dashboard)
- [Model B — PE-PhenoRisk (doctor dashboard)](#model-b--pe-phenorisk-doctor-dashboard)
- [Dashboards](#dashboards)
- [Getting started](#getting-started)
- [Training the models](#training-the-models)
- [Running the apps](#running-the-apps)
- [Model artifacts](#model-artifacts)
- [Evaluation](#evaluation)
- [Clinical disclaimer](#clinical-disclaimer)
- [Team](#team)
- [License](#license)

---

## Why two models

A single classifier cannot serve both audiences well.

| Layer | Audience | Goal | Data it needs |
| --- | --- | --- | --- |
| **Model A** | Patient / community screening | Catch high-risk cases early (high recall) | Age, BP, BMI, blood sugar, heart rate, medical history, environmental context |
| **Model B** | Clinician | Confirm risk and explain *which phenotype* a patient resembles | Labs and placental biomarkers (PlGF/sFlt, sEng, PP-13, cystatin C, creatinine, and related vitals) |

Model A is optimized so fewer high-risk pregnancies are missed. Model B is optimized so a clinician can see a phenotype-driven probability grounded in lab confirmation.

---

## System architecture

```text
                    ┌─────────────────────────────────────┐
  Everyday vitals   │  Model A — BioFusion-NN             │
  + history         │  Supervised MLP (recall-optimized)  │──▶ Risk score
  + environment     │  Streamlit: app/app.py              │    + action plan
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  Labs +            │  Model B — PE-PhenoRisk             │
  biomarkers        │  K-Means phenotypes → Random Forest │──▶ Phenotype
  + clinical flags  │  Streamlit: app/app_doc.py          │    + lab-confirmed risk
                    └─────────────────────────────────────┘
```

**Model A pipeline**

1. Median-impute missing numeric values.
2. Map `Risk Level` to a binary target (`High` = 1, otherwise 0).
3. Engineer Mean Arterial Pressure: `MAP = (Systolic + 2 × Diastolic) / 3`.
4. Add environmental context features (heat, air quality, access to care), with a domain rule that hypertensive patients are treated as high heat exposure.
5. Train an `MLPClassifier` inside a `StandardScaler` pipeline. Hyperparameters are chosen by 5-fold stratified grid search with **recall** as the scoring metric.
6. Explain predictions with a Random Forest surrogate and SHAP.

**Model B pipeline**

1. Cluster patients on seven clinically weighted biomarkers using K-Means (`k = 2`).
2. Score cluster centers with biomarker weights (PlGF/sFlt weighted highest) and label the higher-scoring cluster as the high-risk phenotype.
3. Train a class-balanced Random Forest to predict that phenotype from the full numeric + binary feature set.
4. Report ROC-AUC, a classification report, and feature importances.

---

## Repository layout

```text
two_stage_preeclampsia_AI_model/
├── app/
│   ├── app.py          # Patient dashboard (Model A)
│   └── app_doc.py      # Doctor / lab dashboard (Model B)
├── data/
│   ├── Dataset - Updated.csv   # Model A training data (~1,205 rows)
│   └── preeclampsia.csv        # Model B training data (~400 rows)
├── work2.py            # Train Model A (BioFusion-NN)
├── work.py             # Train Model B (PE-PhenoRisk)
├── requirements.txt
├── LICENSE             # MIT
└── README.md
```

Trained `.pkl` files are **not** committed. You generate them by running the training scripts, then place them where the dashboards look for them (see [Model artifacts](#model-artifacts)).

---

## Datasets

### Model A — `data/Dataset - Updated.csv`

Maternal health screening table (~1,205 records). Used for supervised high/low risk classification.

| Column | Role |
| --- | --- |
| `Age` | Demographic |
| `Systolic BP`, `Diastolic` | Blood pressure (mmHg) |
| `BS` | Blood sugar (mmol/L) |
| `Body Temp` | Body temperature (°F) |
| `BMI` | Body mass index |
| `Heart Rate` | Heart rate (bpm) |
| `Previous Complications` | Binary history flag |
| `Preexisting Diabetes` | Binary history flag |
| `Gestational Diabetes` | Binary history flag |
| `Mental Health` | Binary history flag |
| `Risk Level` | Label: `High` / `Mid` / `Low`. Training maps `High` → 1 and everything else → 0 |

Engineered at train time (not stored in the CSV):

- `MAP` — mean arterial pressure
- `Heat_Exposure`, `Air_Pollution`, `Access_To_Care` — contextual features used to simulate low-resource screening settings

### Model B — `data/preeclampsia.csv`

Clinical and laboratory table (~400 records). Used for unsupervised phenotyping, then supervised phenotype classification. There is **no explicit preeclampsia label** in this file; the target is discovered from clusters.

**Demographic and anthropometric**

`age`, `gest_age`, `height`, `weight`, `bmi`

**Vital signs**

`sysbp`, `diabp`

**Laboratory and biomarkers**

`hb`, `pcv`, `tsh`, `platelet`, `creatinine`, `plgf:sflt` (normalized to `plgfsflt` in code), `SEng` / `seng`, `cysC` / `cysc`, `pp_13`, `glycerides`

**Binary clinical flags**

`htn`, `diabetes`, `fam_htn`, `sp_art` (assisted reproduction)

**Lifestyle (present in the CSV; not used as classifier inputs)**

`occupation`, `diet`, `activity`, `sleep`

---

## Model A — BioFusion-NN (patient dashboard)

**Objective:** early high-sensitivity screening from data that can be collected outside a full lab workup.

| Item | Detail |
| --- | --- |
| Algorithm | `sklearn.neural_network.MLPClassifier` in a `Pipeline` with `StandardScaler` |
| Split | 80 / 20, stratified, `random_state=42` |
| Tuning | `GridSearchCV` + `StratifiedKFold(n_splits=5)` |
| Scoring | `recall` (sensitivity) |
| Search space | Hidden layers `(64, 32)`, `(128, 64)`, `(64, 32, 16)`; activations `relu` / `tanh`; `alpha` `1e-4` / `1e-3`; learning rate `0.001` / `0.01` |
| Explainability | Random Forest surrogate + SHAP summary plot |
| Serialized model | `biofusion_model_v1.pkl` |

**Risk bands used at inference**

| Probability | Category | Suggested action |
| --- | --- | --- |
| `< 0.30` | Low | Routine antenatal care |
| `0.30 – 0.69` | Moderate | Increase monitoring (e.g. weekly) |
| `≥ 0.70` | High | Immediate clinical referral |

---

## Model B — PE-PhenoRisk (doctor dashboard)

**Objective:** find latent preeclampsia phenotypes, then predict which phenotype a new lab profile belongs to.

### Step 1 — Unsupervised phenotyping

K-Means (`k = 2`, `n_init=100`, `random_state=42`) on:

`sysbp`, `diabp`, `plgfsflt`, `creatinine`, `seng`, `cysc`, `pp_13`

Preprocessing: median imputation + z-score scaling. Cluster quality is reported with a silhouette score.

### Step 2 — Label the high-risk phenotype

Cluster centers are dotted with clinical weights. PlGF/sFlt is weighted `1.3`, creatinine `1.1`, remaining features `1.0`. The cluster with the larger score is labeled **high-risk phenotype (`1`)**.

A sanity check prints mean `sysbp`, `diabp`, `plgfsflt`, and `creatinine` by phenotype.

### Step 3 — Supervised classifier

| Item | Detail |
| --- | --- |
| Algorithm | `RandomForestClassifier` (`n_estimators=300`, `max_depth=10`, `class_weight='balanced'`) |
| Inputs | All numeric columns plus the four binary flags |
| Target | `phenotype_group` from Step 2 |
| Split | 80 / 20, stratified |
| Metrics | Accuracy, ROC-AUC, classification report, Gini importances |

Reported ROC-AUC on the held-out set is about **0.96** (see training output when you re-run `work.py`).

The doctor app currently thresholds probability at **0.60**: above that is treated as confirmed high risk.

---

## Dashboards

Both UIs are Streamlit apps.

### Patient dashboard — `app/app.py`

- Sidebar inputs: age, systolic/diastolic BP, blood sugar, temperature, heart rate, BMI, gestational diabetes, preexisting diabetes, mental health history, previous complications, heat exposure, air quality, access to care.
- Computes MAP on the fly.
- Loads `biofusion_model_v1.pkl` from the app folder, project root, current working directory, or a `models/` directory.
- Shows risk band, recommended action, and a simple vitals bar chart.

### Doctor dashboard — `app/app_doc.py`

- Full lab/clinical form: age, gestational age, height, weight, BMI, BP, hemoglobin, PCV, TSH, platelets, creatinine, PlGF/sFlt, sEng, cystatin C, PP-13, glycerides, hypertension, diabetes, family HTN, assisted reproduction.
- Expects **21 features** in this order:

  `age`, `gest_age`, `height`, `weight`, `bmi`, `sysbp`, `diabp`, `hb`, `pcv`, `tsh`, `platelet`, `creatinine`, `plgfsflt`, `SEng`, `cysC`, `pp_13`, `glycerides`, `htn`, `diabetes`, `fam_htn`, `sp_art`

- Loads `models/preeclampsia_model.pkl` and `models/scaler.pkl`.
- Displays lab-confirmed risk probability and a high/low action.

---

## Getting started

### Requirements

- Python 3.9+ recommended
- pip

### Install

From the repository root:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists Streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, and plotly.

Training Model A also needs **SHAP**. Serialization uses **joblib** (usually installed with scikit-learn; install it explicitly if needed):

```bash
pip install shap joblib
```

---

## Training the models

Run these from the **repository root** so relative data paths resolve.

### Model A

```bash
python work2.py
```

This reads `data/Dataset - Updated.csv`, runs grid search, prints a classification report and ROC-AUC, shows a precision-recall curve and a SHAP summary, and writes `biofusion_model_v1.pkl` in the current directory.

Grid search can take several minutes.

### Model B

```bash
python work.py
```

`work.py` currently loads `preeclampsia.csv` from the working directory. Either copy the file first or point the script at `data/preeclampsia.csv`:

```bash
copy data\preeclampsia.csv preeclampsia.csv
python work.py
```

On success it writes:

- `preeclampsia_phenotype_model.pkl`
- `cluster_imputer.pkl`
- `cluster_scaler.pkl`
- `classifier_imputer.pkl`
- `feature_columns.pkl`

---

## Running the apps

Train (or obtain) the pickle files first. Then:

```bash
streamlit run app/app.py
```

```bash
streamlit run app/app_doc.py
```

The browser should open to `http://localhost:8501`.

---

## Model artifacts

| File | Produced by | Consumed by |
| --- | --- | --- |
| `biofusion_model_v1.pkl` | `work2.py` | `app/app.py` — searched in `app/`, repo root, cwd, and `models/` |
| `preeclampsia_phenotype_model.pkl` | `work.py` | Training artifact for Model B |
| `cluster_imputer.pkl`, `cluster_scaler.pkl`, `classifier_imputer.pkl`, `feature_columns.pkl` | `work.py` | Preprocessing objects saved with Model B |
| `models/preeclampsia_model.pkl` | Place / rename after training | `app/app_doc.py` |
| `models/scaler.pkl` | Place / rename after training | `app/app_doc.py` |

The doctor app looks specifically under `models/` for `preeclampsia_model.pkl` and `scaler.pkl`. After training Model B, create that folder and copy the classifier (and a fitted scaler, if you export one) to those names, or update the paths in `app/app_doc.py`.

---

## Evaluation

| Model | Primary goal | What to look at | Notes |
| --- | --- | --- | --- |
| Model A | Early screening | Recall, precision-recall AUC, ROC-AUC, classification report | Grid search maximizes recall so high-risk cases are less likely to be missed |
| Model B | Clinical stratification | ROC-AUC (~0.96), confusion matrix, top feature importances | Target is the discovered phenotype, not a raw chart label |

Re-run the training scripts to reproduce metrics on your machine; they depend on the exact scikit-learn version and the train/test split seed (`42`).

---

## Clinical disclaimer

This software is a **research and educational prototype**. It is **not** FDA/CE cleared, **not** a medical device, and **not** a substitute for clinical assessment, laboratory testing, or professional guidelines.

Do not use these scores as the sole basis for diagnosis, admission, or treatment. Any “action plan” text in the dashboards is illustrative.

---

## Team

BioFusion Hackathon project.

| Person | Role |
| --- | --- |
| **Rashmi Paboda** | Project concept and system design; dual-model architecture; Model A (BioFusion-NN); Model B (PE-PhenoRisk); dataset selection and preprocessing; feature engineering and evaluation; explainability (SHAP, surrogate model); dashboard logic and deployment preparation; documentation |
| **Easha Sameekshika** | Team lead; project documentation |
| **Danul Renuja** | Model A changes and Model A documentation |

**Primary author:** Rashmi Paboda — Computer Science & Engineering, maternal health AI research.

Repository: [github.com/rash200319/two_stage_preeclampsia_AI_model](https://github.com/rash200319/two_stage_preeclampsia_AI_model)

If this work is useful, please star the repository.

---

## License

This project is released under the [MIT License](LICENSE). Copyright (c) 2026 R.P.M. Vithanage.
