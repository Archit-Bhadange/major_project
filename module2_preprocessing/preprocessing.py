"""
MODULE 2 — DATA PREPROCESSING
==============================
Owner : Member 2
Input : outputs/train_with_RUL.csv (from Module 1)
Output: outputs/X_train.npy, outputs/X_test.npy,
        outputs/y_train_class.npy, outputs/y_test_class.npy,
        outputs/y_train_rul.npy, outputs/y_test_rul.npy,
        outputs/X_train_seq.npy, outputs/X_test_seq.npy,
        outputs/y_train_seq.npy, outputs/y_test_seq.npy,
        models/scaler.pkl

Run this file from the project root:
    python module2_preprocessing/preprocessing.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("MODULE 2 — DATA PREPROCESSING")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1 — LOAD DATA FROM MODULE 1
# ─────────────────────────────────────────
print("\n[1/8] Loading data from Module 1...")

train_df = pd.read_csv('outputs/train_with_RUL.csv')

# Load useful sensors identified in Module 1
with open('outputs/useful_sensors.json', 'r') as f:
    useful_sensors = json.load(f)

print(f"✅ Loaded {len(train_df)} rows")
print(f"✅ Using {len(useful_sensors)} useful sensors: {useful_sensors}")

# ─────────────────────────────────────────
# STEP 2 — DROP CONSTANT SENSORS
# ─────────────────────────────────────────
print("\n[2/8] Dropping constant sensors...")

# Keep only useful sensors + metadata columns
keep_cols = ['engine_id', 'cycle',
             'setting_1', 'setting_2', 'setting_3'] + \
             useful_sensors + ['RUL']

train_df = train_df[keep_cols]
print(f"✅ Kept {len(keep_cols)} columns")
print(f"✅ Dropped 7 constant sensors")

# ─────────────────────────────────────────
# STEP 3 — CREATE BINARY CLASSIFICATION LABEL
# ─────────────────────────────────────────
print("\n[3/8] Creating binary classification label...")

# Rule: If RUL <= 30 cycles → engine will fail soon → label = 1
#       If RUL >  30 cycles → engine is healthy      → label = 0
FAILURE_THRESHOLD = 30

train_df['label'] = (train_df['RUL'] <= FAILURE_THRESHOLD).astype(int)

failure_count = train_df['label'].sum()
healthy_count = len(train_df) - failure_count
print(f"✅ Failure label (1): {failure_count} rows ({failure_count/len(train_df)*100:.1f}%)")
print(f"✅ Healthy label (0): {healthy_count} rows ({healthy_count/len(train_df)*100:.1f}%)")
print(f"⚠️  Class imbalance ratio: 1:{healthy_count//failure_count}")

# ─────────────────────────────────────────
# STEP 4 — DEFINE FEATURES
# ─────────────────────────────────────────
print("\n[4/8] Defining feature columns...")

# Features = operational settings + useful sensors
feature_cols = ['setting_1', 'setting_2', 'setting_3'] + useful_sensors
print(f"✅ Total features: {len(feature_cols)}")
print(f"   {feature_cols}")

# ─────────────────────────────────────────
# STEP 5 — NORMALISE FEATURES (MinMaxScaler)
# ─────────────────────────────────────────
print("\n[5/8] Normalising features (MinMaxScaler 0 to 1)...")

scaler = MinMaxScaler()

# Fit scaler on training features and transform
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# IMPORTANT: Save scaler — must use same scaler during prediction
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
print(f"✅ Scaler fitted and saved to models/scaler.pkl")
print(f"✅ All features normalised to 0-1 range")

# ─────────────────────────────────────────
# STEP 6 — TRAIN TEST SPLIT
# ─────────────────────────────────────────
print("\n[6/8] Splitting into train and test sets...")

X = train_df[feature_cols].values
y_class = train_df['label'].values    # For classification (RF, XGBoost)
y_rul = train_df['RUL'].values        # For regression (LSTM)

X_train, X_test, y_train_class, y_test_class, y_train_rul, y_test_rul = \
    train_test_split(
        X, y_class, y_rul,
        test_size=0.2,
        random_state=42,
        stratify=y_class  # Ensure balanced split
    )

print(f"✅ Training set  : {X_train.shape[0]} rows")
print(f"✅ Test set      : {X_test.shape[0]} rows")

# ─────────────────────────────────────────
# STEP 7 — APPLY SMOTE (Fix Class Imbalance)
# ─────────────────────────────────────────
print("\n[7/8] Applying SMOTE to fix class imbalance...")

before_0 = np.sum(y_train_class == 0)
before_1 = np.sum(y_train_class == 1)
print(f"Before SMOTE → Class 0: {before_0}, Class 1: {before_1}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train_class)

after_0 = np.sum(y_train_smote == 0)
after_1 = np.sum(y_train_smote == 1)
print(f"After SMOTE  → Class 0: {after_0}, Class 1: {after_1}")
print(f"✅ Classes are now balanced!")

# ─────────────────────────────────────────
# STEP 8 — CREATE LSTM SEQUENCES
# ─────────────────────────────────────────
print("\n[8/8] Creating sliding window sequences for LSTM...")

SEQUENCE_LENGTH = 30  # 30 consecutive cycles per sequence

def create_sequences(df, feature_cols, sequence_length):
    """
    Creates sliding window sequences for LSTM.

    For each engine:
    - Takes 30 consecutive rows
    - Creates one sequence
    - Slides window by 1 each time

    Example for engine with 100 cycles:
    Window 1: cycles 1-30  → predict RUL at cycle 30
    Window 2: cycles 2-31  → predict RUL at cycle 31
    ...
    Window 71: cycles 71-100 → predict RUL at cycle 100
    """
    sequences = []
    targets = []

    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        engine_features = engine_data[feature_cols].values
        engine_rul = engine_data['RUL'].values

        # Create sliding windows
        for i in range(len(engine_data) - sequence_length + 1):
            seq = engine_features[i:i + sequence_length]
            target = engine_rul[i + sequence_length - 1]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)

# Create sequences from full training data (before train/test split)
# We need engine_id for grouping — use original train_df
X_seq, y_seq = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)

# Split sequences into train/test
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq,
    test_size=0.2,
    random_state=42
)

print(f"✅ Sequence shape     : {X_train_seq.shape}")
print(f"   (samples, timesteps, features) = {X_train_seq.shape}")
print(f"✅ Training sequences : {X_train_seq.shape[0]}")
print(f"✅ Test sequences     : {X_test_seq.shape[0]}")

# ─────────────────────────────────────────
# SAVE ALL OUTPUTS
# ─────────────────────────────────────────
print("\n💾 Saving all preprocessed data...")

os.makedirs('outputs', exist_ok=True)

# ML data (for Random Forest and XGBoost)
np.save('outputs/X_train.npy', X_train_smote)
np.save('outputs/X_test.npy', X_test)
np.save('outputs/y_train_class.npy', y_train_smote)
np.save('outputs/y_test_class.npy', y_test_class)
np.save('outputs/y_train_rul.npy', y_train_rul)
np.save('outputs/y_test_rul.npy', y_test_rul)

# LSTM sequence data
np.save('outputs/X_train_seq.npy', X_train_seq)
np.save('outputs/X_test_seq.npy', X_test_seq)
np.save('outputs/y_train_seq.npy', y_train_seq)
np.save('outputs/y_test_seq.npy', y_test_seq)

# Save feature column names for later use
with open('outputs/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print("✅ Saved: outputs/X_train.npy")
print("✅ Saved: outputs/X_test.npy")
print("✅ Saved: outputs/y_train_class.npy")
print("✅ Saved: outputs/y_test_class.npy")
print("✅ Saved: outputs/y_train_rul.npy")
print("✅ Saved: outputs/y_test_rul.npy")
print("✅ Saved: outputs/X_train_seq.npy")
print("✅ Saved: outputs/X_test_seq.npy")
print("✅ Saved: outputs/y_train_seq.npy")
print("✅ Saved: outputs/y_test_seq.npy")
print("✅ Saved: outputs/feature_cols.json")
print("✅ Saved: models/scaler.pkl")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("MODULE 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"✅ Constant sensors dropped : 7")
print(f"✅ Features used            : {len(feature_cols)}")
print(f"✅ Normalisation            : MinMaxScaler (0-1)")
print(f"✅ Class imbalance fixed    : SMOTE applied")
print(f"✅ ML train samples         : {X_train_smote.shape[0]}")
print(f"✅ ML test samples          : {X_test.shape[0]}")
print(f"✅ LSTM train sequences     : {X_train_seq.shape[0]}")
print(f"✅ LSTM sequence shape      : {X_train_seq.shape}")
print(f"✅ Scaler saved             : models/scaler.pkl")
print("\n👉 Next Step: Run Module 3 — ML Models")
print("=" * 60)