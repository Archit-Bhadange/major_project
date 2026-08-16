"""
MODULE 3 — MACHINE LEARNING MODELS
====================================
Owner : Member 3
Input : outputs/ folder (from Module 2)
Output: models/random_forest.pkl
        models/xgboost_model.pkl
        outputs/ml_results.json
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

print("=" * 60)
print("MODULE 3 — MACHINE LEARNING MODELS")
print("=" * 60)

# 1. Load Preprocessed Data
print("\n[1/6] Loading preprocessed data from Module 2...")
X_train = np.load('outputs/X_train.npy')
X_test  = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train_class.npy')
y_test  = np.load('outputs/y_test_class.npy')

with open('outputs/feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

print(f"✅ X_train shape : {X_train.shape}")
print(f"✅ X_test shape  : {X_test.shape}")
print(f"✅ Features      : {len(feature_cols)}")

# 2. Train Random Forest
print("\n[2/6] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
print("✅ Random Forest trained successfully!")

# 3. Train XGBoost
print("\n[3/6] Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
print("✅ XGBoost trained successfully!")

# 4. Evaluate Models
print("\n[4/6] Evaluating metrics...")
def evaluate_model(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n  {name} Results:")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc*100:.2f}%")

    return {
        'accuracy': round(acc*100, 2),
        'precision': round(prec*100, 2),
        'recall': round(rec*100, 2),
        'f1_score': round(f1*100, 2),
        'roc_auc': round(auc*100, 2)
    }

rf_results  = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
xgb_results = evaluate_model("XGBoost",       y_test, xgb_pred, xgb_prob)

best_model_name = "Random Forest" if rf_results['f1_score'] >= xgb_results['f1_score'] else "XGBoost"
print(f"\n🏆 Best Model (by F1 Score): {best_model_name}")

results = {
    'random_forest': rf_results,
    'xgboost': xgb_results,
    'best_model': best_model_name
}
with open('outputs/ml_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 5. Save Models
print("\n[5/6] Saving trained model artifacts...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model,  'models/random_forest.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("✅ Models saved to models/")

# 6. SHAP Plots & Visualizations
print("\n[6/6] Generating evaluation plots...")

# Plot Confusion Matrices
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Failure'], yticklabels=['Healthy', 'Failure'])
plt.title('Random Forest — Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/plot_rf_confusion.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Healthy', 'Failure'], yticklabels=['Healthy', 'Failure'])
plt.title('XGBoost — Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/plot_xgb_confusion.png', dpi=150)
plt.close()

# SHAP Analysis
try:
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_vals_rf = explainer_rf.shap_values(X_test[:200])
    shap_plot_rf = shap_vals_rf[1] if isinstance(shap_vals_rf, list) else shap_vals_rf
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_plot_rf, X_test[:200], feature_names=feature_cols, show=False)
    plt.title('SHAP Feature Importance — Random Forest')
    plt.tight_layout()
    plt.savefig('outputs/plot_shap_rf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP plots saved!")
except Exception as e:
    print(f"⚠️ SHAP generation skipped: {e}")

print("\n" + "=" * 60)
print("MODULE 3 COMPLETE")
print("=" * 60)