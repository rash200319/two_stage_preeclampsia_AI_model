import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('preeclampsia.csv', header=None)

# Assign column names based on dataset structure
columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 'hb', 
           'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 'SEng', 'cysC', 
           'pp_13', 'glycerides', 'htn', 'diabetes', 'fam_htn', 'sp_art', 
           'occupation', 'diet', 'activity', 'sleep']
df.columns = columns

# Convert all columns to numeric (handle any parsing issues)
numeric_columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 
                   'hb', 'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 
                   'SEng', 'cysC', 'pp_13', 'glycerides']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Create preeclampsia risk label using key risk factors
# High risk if: high BP, high plgf:sflt ratio, proteinuria indicators, family history
df['preeclampsia_risk'] = 0

# Risk scoring based on clinical factors
risk_score = 0
df['risk_score'] = (
    (df['sysbp'] > 140).astype(int) * 3 +
    (df['diabp'] > 90).astype(int) * 3 +
    (df['plgfsflt'] > 100).astype(int) * 2 +  # Abnormal PlGF/sFlt-1 ratio
    (df['SEng'] > 20).astype(int) * 2 +
    (df['creatinine'] > 1.1).astype(int) * 2 +
    (df['htn'] == 1).astype(int) * 2 +
    (df['fam_htn'] == 1).astype(int) * 1 +
    (df['diabetes'] == 1).astype(int) * 1 +
    (df['bmi'] > 30).astype(int) * 1 +
    (df['age'] > 35).astype(int) * 1
)

# Create binary target: high risk (1) vs low risk (0)
df.loc[df['risk_score'] >= 5, 'preeclampsia_risk'] = 1

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['preeclampsia_risk'].value_counts())
print("\nRisk score statistics:")
print(df['risk_score'].describe())

# Prepare features and target
X = df[numeric_columns + ['htn', 'diabetes', 'fam_htn', 'sp_art']]
y = df['preeclampsia_risk']

# With this clean version:
X = df[numeric_columns + ['htn', 'diabetes', 'fam_htn', 'sp_art']].copy()  # ✅ Fixed!
binary_cols = ['htn', 'diabetes', 'fam_htn', 'sp_art']
X.loc[:, binary_cols] = X[binary_cols].apply(pd.to_numeric, errors='coerce')  # ✅ Safe
# Fill numeric missing values with column medians

X = X.fillna(X.median(numeric_only=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model (good for medical data with mixed features)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                 random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nModel Performance:")
print("Accuracy:", rf_model.score(X_test_scaled, y_test))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# Save model and scaler for competition submission
import joblib
joblib.dump(rf_model, 'preeclampsia_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(columns, 'feature_columns.pkl')

print("\nModel saved as 'preeclampsia_model.pkl'")
print("Ready for competition submission!")

# Prediction function for new data
def predict_risk(new_data):
    """
    Predict preeclampsia risk for new patient data
    new_data: dictionary with feature names as keys
    """
    new_df = pd.DataFrame([new_data])
    new_df[numeric_columns] = new_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # Ensure binary/categorical features are numeric for consistent preprocessing
    binary_cols = ['htn', 'diabetes', 'fam_htn', 'sp_art']
    new_df[binary_cols] = new_df[binary_cols].apply(pd.to_numeric, errors='coerce')
    new_df = new_df.fillna(new_df.median(numeric_only=True))
    
    new_scaled = scaler.transform(new_df[X.columns])
    risk = rf_model.predict_proba(new_scaled)[0][1]
    
    return {
        'risk_probability': risk,
        'risk_category': 'High' if risk > 0.5 else 'Low'
    }

print("\nPrediction function ready for new patients!")
