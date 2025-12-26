import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer 

# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------
# Load the dataset
try:
    df = pd.read_csv('preeclampsia.csv', header=None)
except FileNotFoundError:
    print("Error: 'preeclampsia.csv' not found. Please check the file path.")
    exit()

#the dataset columns are wrong so we switch those up
df = df.rename(columns={'sysbp': 'diabp', 'diabp': 'sysbp'})

#standardize columns 
df.columns = df.columns.str.strip().str.lower().str.replace(":", "", regex=False)

# Assign column names based on dataset structure
expected_columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 'hb', 
           'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 'seng', 'cysc', 
           'pp_13', 'glycerides', 'htn', 'diabetes', 'fam_htn', 'sp_art', 
           'occupation', 'diet', 'activity', 'sleep']

#check if any columns are missing( error handling)
missing = set(expected_columns) - set(df.columns)
if missing:
    raise ValueError(f" Missing required columns: {missing}")
df= df[expected_columns]

# Convert all columns to numeric (handle any parsing issues)
numeric_columns = ['age', 'gest_age', 'height', 'weight', 'bmi', 'sysbp', 'diabp', 
                   'hb', 'pcv', 'tsh', 'platelet', 'creatinine', 'plgfsflt', 
                   'seng', 'cysc', 'pp_13', 'glycerides']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Handle binary columns safely
binary_cols = ['htn', 'diabetes', 'fam_htn', 'sp_art']
df[binary_cols] = df[binary_cols].apply(pd.to_numeric, errors='coerce')


# 2. UNSUPERVISED LABELING (The Scientific Fix)
# ---------------------------------------------------------
print("--- Starting Unsupervised Label Generation ---")

# Select features for clustering (The medical markers)
cluster_cols = ['sysbp', 'diabp', 'plgfsflt', 'creatinine', 'SEng', 'cysC', 'pp_13']
X_cluster = df[cluster_cols].copy()

# A. Impute Missing Values (The Fix for 'ValueError: Input X contains NaN')
imputer_cluster = SimpleImputer(strategy='median')
X_cluster_imputed = imputer_cluster.fit_transform(X_cluster)

# B. Scale the data (Clustering is sensitive to scale)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster_imputed)

# C. Let AI find 2 natural groups (Cluster 0 and Cluster 1)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['cluster_label'] = kmeans.fit_predict(X_cluster_scaled)

# D. Identify which cluster is "High Risk"
# We compare the mean Systolic BP of both groups. The group with higher BP is the Risk group.
cluster_0_bp = df[df['cluster_label'] == 0]['sysbp'].mean()
cluster_1_bp = df[df['cluster_label'] == 1]['sysbp'].mean()

print(f"Cluster 0 Mean BP: {cluster_0_bp:.1f}")
print(f"Cluster 1 Mean BP: {cluster_1_bp:.1f}")

if cluster_1_bp > cluster_0_bp:
    high_risk_label = 1
    print(">> AI identified Cluster 1 as the HIGH RISK group.")
else:
    high_risk_label = 0
    # Flip labels so 1 is always High Risk for consistency
    df['cluster_label'] = 1 - df['cluster_label'] 
    print(">> AI identified Cluster 0 as the HIGH RISK group (Labels flipped).")

# Assign this as your target variable
df['preeclampsia_risk'] = df['cluster_label']

print("\nTarget distribution (AI Discovery):")
print(df['preeclampsia_risk'].value_counts())


# 3. SUPERVISED TRAINING (Random Forest)
# ---------------------------------------------------------
print("\n--- Starting Supervised Model Training ---")

# Prepare features and target
X = df[numeric_columns + binary_cols].copy()
y = df['preeclampsia_risk']

# Fill numeric missing values with column medians for the supervised training set
X = X.fillna(X.median(numeric_only=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                 random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)


# 4. EVALUATION
# ---------------------------------------------------------
# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

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

# 5. SAVE ARTIFACTS
# ---------------------------------------------------------
joblib.dump(rf_model, 'preeclampsia_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(columns, 'feature_columns.pkl')

print("\nModel saved as 'preeclampsia_model.pkl'")
print("Ready for competition submission!")
