"""
MODULE 4 — LSTM DEEP LEARNING MODEL
=====================================
Owner : Member 4
Input : outputs/ folder (from Module 2)
Output: models/lstm_model.h5
        outputs/lstm_results.json
        outputs/plot_lstm_loss.png
        outputs/plot_lstm_predictions.png
        outputs/plot_all_models_comparison.png

Run from project root:
    python module4_lstm/lstm_model.py
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("MODULE 4 — LSTM DEEP LEARNING MODEL")
print("=" * 60)
print(f"TensorFlow version : {tf.__version__}")
print(f"Running on         : CPU (Ryzen 7 6800H)")

# ─────────────────────────────────────────
# STEP 1 — LOAD SEQUENCE DATA FROM MODULE 2
# ─────────────────────────────────────────
print("\n[1/6] Loading sequence data from Module 2...")

X_train_seq = np.load('outputs/X_train_seq.npy')
X_test_seq  = np.load('outputs/X_test_seq.npy')
y_train_seq = np.load('outputs/y_train_seq.npy')
y_test_seq  = np.load('outputs/y_test_seq.npy')

print(f"✅ X_train_seq shape : {X_train_seq.shape}")
print(f"✅ X_test_seq shape  : {X_test_seq.shape}")
print(f"   (samples, timesteps, features)")

# ─────────────────────────────────────────
# STEP 2 — NORMALISE RUL TARGET VALUES
# ─────────────────────────────────────────
print("\n[2/6] Normalising RUL target values...")

# Scale RUL values to 0-1 range for better LSTM training
rul_scaler = MinMaxScaler()
y_train_scaled = rul_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
y_test_scaled  = rul_scaler.transform(y_test_seq.reshape(-1, 1)).flatten()

print(f"✅ RUL range in training: {y_train_seq.min():.0f} to {y_train_seq.max():.0f} cycles")
print(f"✅ Scaled to 0-1 range for training")

# ─────────────────────────────────────────
# STEP 3 — BUILD LSTM ARCHITECTURE
# ─────────────────────────────────────────
print("\n[3/6] Building LSTM architecture...")

# Get dimensions from data
timesteps = X_train_seq.shape[1]   # 30 cycles
features  = X_train_seq.shape[2]   # 18 features

model = Sequential([

    # First LSTM layer
    # units=64 → 64 memory cells
    # return_sequences=True → passes output to next LSTM layer
    LSTM(units=64,
         return_sequences=True,
         input_shape=(timesteps, features)),
    Dropout(0.2),  # Randomly drop 20% connections to prevent overfitting

    # Second LSTM layer
    # return_sequences=False → only final output passed forward
    LSTM(units=32,
         return_sequences=False),
    Dropout(0.2),

    # Dense hidden layer
    Dense(units=16, activation='relu'),

    # Output layer — single value = predicted RUL
    Dense(units=1, activation='linear')
])

# Compile model
model.compile(
    optimizer='adam',       # Adam optimizer — best for LSTM
    loss='mse',             # Mean Squared Error loss
    metrics=['mae']         # Track Mean Absolute Error
)

# Print model summary
model.summary()

print(f"\n✅ LSTM architecture built!")
print(f"   Input shape  : ({timesteps}, {features})")
print(f"   Output shape : (1,) → single RUL value")

# ─────────────────────────────────────────
# STEP 4 — TRAIN LSTM
# ─────────────────────────────────────────
print("\n[4/6] Training LSTM model...")
print("      (This will take 10-15 minutes on CPU — please wait)")
print("      Watch loss decreasing each epoch — that means model is learning!")

os.makedirs('models', exist_ok=True)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,          # Stop if no improvement for 10 epochs
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'models/lstm_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train the model
history = model.fit(
    X_train_seq, y_train_scaled,
    epochs=50,              # Maximum 50 training rounds
    batch_size=64,          # Process 64 sequences at once
    validation_split=0.1,   # Use 10% of training data for validation
    callbacks=[early_stopping, model_checkpoint],
    verbose=1               # Show progress each epoch
)

print("\n✅ LSTM training complete!")

# ─────────────────────────────────────────
# STEP 5 — EVALUATE LSTM
# ─────────────────────────────────────────
print("\n[5/6] Evaluating LSTM...")

# Make predictions
y_pred_scaled = model.predict(X_test_seq, verbose=0)

# Convert back to original RUL scale
y_pred = rul_scaler.inverse_transform(y_pred_scaled).flatten()
y_actual = y_test_seq

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)

print(f"\n  LSTM Results:")
print(f"  RMSE : {rmse:.2f} cycles")
print(f"  MAE  : {mae:.2f} cycles")
print(f"\n  Interpretation:")
print(f"  → Model predictions are off by ~{mae:.0f} cycles on average")
print(f"  → Lower is better (0 = perfect prediction)")

# Save results
lstm_results = {
    'rmse': round(float(rmse), 2),
    'mae': round(float(mae), 2),
    'epochs_trained': len(history.history['loss'])
}

with open('outputs/lstm_results.json', 'w') as f:
    json.dump(lstm_results, f, indent=2)

# ─────────────────────────────────────────
# STEP 6 — VISUALISATIONS
# ─────────────────────────────────────────
print("\n[6/6] Generating plots...")

# ── Plot 1: Training Loss Curve ──
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],
         label='Training Loss', color='steelblue')
plt.plot(history.history['val_loss'],
         label='Validation Loss', color='coral')
plt.title('LSTM Training — Loss Curve\n(Decreasing loss = model is learning)',
          fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plot_lstm_loss.png', dpi=150)
plt.close()
print("✅ Saved: plot_lstm_loss.png")

# ── Plot 2: Actual vs Predicted RUL ──
plt.figure(figsize=(12, 5))

# Show first 200 test predictions
n = min(200, len(y_actual))
plt.plot(y_actual[:n],
         label='Actual RUL',    color='steelblue', linewidth=1.5)
plt.plot(y_pred[:n],
         label='Predicted RUL', color='coral',
         linewidth=1.5, linestyle='--')
plt.title('LSTM — Actual vs Predicted RUL\n(Closer the lines = better prediction)',
          fontsize=14)
plt.xlabel('Test Sample')
plt.ylabel('Remaining Useful Life (cycles)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plot_lstm_predictions.png', dpi=150)
plt.close()
print("✅ Saved: plot_lstm_predictions.png")

# ── Plot 3: All Models Comparison ──
# Load ML results from Module 3
with open('outputs/ml_results.json', 'r') as f:
    ml_results = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Complete Model Comparison', fontsize=16)

# Left: ML Models comparison (Classification metrics)
models_ml = ['Random Forest', 'XGBoost']
f1_scores  = [ml_results['random_forest']['f1_score'],
              ml_results['xgboost']['f1_score']]
roc_scores = [ml_results['random_forest']['roc_auc'],
              ml_results['xgboost']['roc_auc']]

x = np.arange(len(models_ml))
width = 0.35
axes[0].bar(x - width/2, f1_scores,  width,
            label='F1 Score', color='steelblue')
axes[0].bar(x + width/2, roc_scores, width,
            label='ROC-AUC',  color='coral')
axes[0].set_title('ML Models — Classification')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_ml)
axes[0].set_ylabel('Score (%)')
axes[0].set_ylim(0, 110)
axes[0].legend()
for i, (f1, roc) in enumerate(zip(f1_scores, roc_scores)):
    axes[0].text(i-width/2, f1+0.5,  f'{f1}%',  ha='center', fontsize=9)
    axes[0].text(i+width/2, roc+0.5, f'{roc}%', ha='center', fontsize=9)

# Right: LSTM metrics (Regression)
axes[1].bar(['RMSE', 'MAE'],
            [rmse, mae],
            color=['steelblue', 'coral'],
            width=0.4)
axes[1].set_title('LSTM — RUL Prediction')
axes[1].set_ylabel('Error (cycles)')
axes[1].text(0, rmse+0.5, f'{rmse:.2f}', ha='center', fontsize=11)
axes[1].text(1, mae+0.5,  f'{mae:.2f}',  ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/plot_all_models_comparison.png', dpi=150)
plt.close()
print("✅ Saved: plot_all_models_comparison.png")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("MODULE 4 COMPLETE — SUMMARY")
print("=" * 60)
print(f"\n  LSTM RUL Prediction:")
print(f"  RMSE           : {rmse:.2f} cycles")
print(f"  MAE            : {mae:.2f} cycles")
print(f"  Epochs trained : {len(history.history['loss'])}")
print(f"\n  ML Models (from Module 3):")
print(f"  Random Forest F1  : {ml_results['random_forest']['f1_score']}%")
print(f"  XGBoost F1        : {ml_results['xgboost']['f1_score']}%")
print(f"\n✅ LSTM model saved  : models/lstm_best.h5")
print(f"✅ Results saved     : outputs/lstm_results.json")
print(f"✅ Plots saved       : outputs/")
print("\n👉 Next Step: Run Module 5 — Dashboard")
print("=" * 60)