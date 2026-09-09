"""
MODULE 3 — MACHINE LEARNING CLASSIFICATION, SCALER, METRICS & EXPLAINABILITY
=============================================================================
Train models, fit scaler on feature columns, print comprehensive metrics 
(Accuracy, Precision, Recall, F1-Score, ROC-AUC), and generate synchronized artifacts.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Load Feature Columns
print("📦 Loading feature column configuration...")
with open('outputs/feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

# 2. Load Processed Dataset
print("📄 Loading outputs/train_with_RUL.csv...")
if not os.path.exists('outputs/train_with_RUL.csv'):
    raise FileNotFoundError("❌ outputs/train_with_RUL.csv not found. Execute Module 2 first.")

df = pd.read_csv('outputs/train_with_RUL.csv')

if 'label_binary' in df.columns:
    y = df['label_binary'].values
elif 'RUL' in df.columns:
    y = (df['RUL'] <= 30).astype(int).values
else:
    raise KeyError("❌ Target label ('label_binary' or 'RUL') not found in dataset.")

X = df[feature_cols]

# 3. Fit and Export Scaler exclusively on feature_cols
print("📏 Fitting StandardScaler on feature columns...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
joblib.dump(scaler, 'models/scaler.pkl')

# 4. Perform Stratified Train-Test Split (80/20)
print("✂️ Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv('outputs/X_train.csv', index=False)
pd.DataFrame({'target': y_train}).to_csv('outputs/y_train.csv', index=False)
X_test.to_csv('outputs/X_test.csv', index=False)
pd.DataFrame({'target': y_test}).to_csv('outputs/y_test.csv', index=False)

# 5. Apply SMOTE
print("⚖️ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Helper Function for Confusion Matrix Plot
def save_confusion_matrix(y_true, y_pred, model_name, filename):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Healthy (0)', 'Risk (1)'],
                yticklabels=['Healthy (0)', 'Risk (1)'])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=12, pad=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', dpi=150, bbox_inches='tight')
    plt.close('all')

# 6. Train & Evaluate Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_prob_rf)

joblib.dump(rf_model, 'models/random_forest.pkl')
save_confusion_matrix(y_test, y_pred_rf, 'Random Forest', 'plot_cm_rf.png')

# 7. Train & Evaluate XGBoost
print("\n⚡ Training XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_roc_auc = roc_auc_score(y_test, y_prob_xgb)

joblib.dump(xgb_model, 'models/xgboost_model.pkl')
save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost', 'plot_cm_xgb.png')

# 8. Print Combined Metrics Table
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_acc, xgb_acc],
    'Precision': [rf_precision, xgb_precision],
    'Recall': [rf_recall, xgb_recall],
    'F1-Score': [rf_f1, xgb_f1],
    'ROC-AUC Score': [rf_roc_auc, xgb_roc_auc]
})

print("\n" + "="*70)
print("📊 COMPREHENSIVE MODEL EVALUATION METRICS (CLASS 1 - FAILURE RISK)")
print("="*70)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*70 + "\n")

# Save metrics to JSON for optional dashboard display
metrics_df.to_json('outputs/model_metrics.json', orient='records', indent=4)

# 9. Generate SHAP Plot — Random Forest
print("🔍 Generating SHAP Plot for Random Forest...")
try:
    plt.close('all')
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_explanation_rf = explainer_rf(X_test.iloc[:200])
    shap_vals_rf = shap_explanation_rf.values[:, :, 1] if len(shap_explanation_rf.shape) == 3 else shap_explanation_rf.values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_rf, X_test.iloc[:200], feature_names=feature_cols, plot_type="dot", show=False)
    plt.title('SHAP Feature Importance — Random Forest', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('outputs/plot_shap_rf.png', dpi=150, bbox_inches='tight')
    plt.close('all')
except Exception as e:
    print(f"❌ Random Forest SHAP Error: {e}")

# 10. Generate SHAP Plot — XGBoost
print("🔍 Generating SHAP Plot for XGBoost...")
try:
    plt.close('all')
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_explanation_xgb = explainer_xgb(X_test.iloc[:200])
    shap_vals_xgb = shap_explanation_xgb.values[:, :, 1] if len(shap_explanation_xgb.shape) == 3 else shap_explanation_xgb.values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_xgb, X_test.iloc[:200], feature_names=feature_cols, plot_type="dot", show=False)
    plt.title('SHAP Feature Importance — XGBoost', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('outputs/plot_shap_xgb.png', dpi=150, bbox_inches='tight')
    plt.close('all')
except Exception as e:
    print(f"❌ XGBoost SHAP Error: {e}")

print("\n🚀 Execution complete! All metrics, models, scaler, and SHAP plots are updated.")