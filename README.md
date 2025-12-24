# Maternal Care AI — Preeclampsia Risk Prediction ✅

**Short description**

This repository contains a simple machine learning pipeline to predict preeclampsia risk from maternal clinical data using a Random Forest classifier. It's intended for educational and research use — not for clinical decision-making.

---

## 🔧 Contents

- `work.py` — Main script: data preprocessing, model training, evaluation, saving model and scaler, and `predict_risk` helper.
- `app.py` — (Optional) small app entrypoint (if present).
- `preeclampsia.csv` and `Dataset - Updated.csv` — Provided datasets (check contents before use).
- `work.ipynb`, `work2.ipynb` — Notebooks for exploration and experiments.
- Saved artifacts (generated after running `work.py`): `preeclampsia_model.pkl`, `scaler.pkl`, `feature_columns.pkl`.
- `requirements.txt` — Python dependencies.

---

## ⚙️ Requirements

- Python 3.8+ recommended
- Install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

(If you don't have `requirements.txt` or want to install manually: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.)

---

## 🚀 Quick start

1. Prepare and inspect your dataset(s) — ensure columns align with `work.py` expected names.
2. Run training & evaluation:

```bash
python work.py
```

You should see training metrics, ROC AUC, and plots. The trained model and scaler will be saved as `.pkl` files.

---

## Usage — programmatic predictions 💡

Load the saved model and make predictions for new patients (keys must match feature names saved in `feature_columns.pkl`):

```python
from joblib import load
model = load('preeclampsia_model.pkl')
scaler = load('scaler.pkl')
cols = load('feature_columns.pkl')

# Example payload (fill with realistic numeric values)
new = {
  'age': 30,
  'gest_age': 20,
  'height': 160,
  'weight': 70,
  'bmi': 27.3,
  'sysbp': 120,
  'diabp': 80,
  'hb': 12.5,
  'pcv': 36,
  'tsh': 2.1,
  'platelet': 250,
  'creatinine': 0.8,
  'plgfsflt': 50,
  'SEng': 10,
  'cysC': 0.9,
  'pp_13': 5,
  'glycerides': 120,
  'htn': 0,
  'diabetes': 0,
  'fam_htn': 0,
  'sp_art': 0
}

# Create DataFrame matching the saved feature order, scale, and predict
import pandas as pd
new_df = pd.DataFrame([new])
new_scaled = scaler.transform(new_df[cols])
prob = model.predict_proba(new_scaled)[0][1]
print('Preeclampsia risk probability:', prob)
```

---

## 🔧 Notes / Troubleshooting

- The scripts expect numeric columns; some binary/categorical columns (e.g., `htn`, `diabetes`, `fam_htn`, `sp_art`) are coerced to numeric before imputing missing values.
- If you see errors related to `median()` or `fillna()`, ensure columns are numeric (the repository contains a fix to coerce binaries and use `median(numeric_only=True)`).
- This model is for experimentation. Do not use in production without clinical validation and regulatory approvals.

---

## 🧪 Notebooks

Open `work.ipynb` and `work2.ipynb` to interactively explore preprocessing, feature importance plots, and other experiments.

---

## License

MIT License — modify as needed.

---

## Contact

For questions, open an issue or email the repository owner.

Thank you! ✨
