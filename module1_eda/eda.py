"""
MODULE 1 — EXPLORATORY DATA ANALYSIS (EDA)
===========================================
Owner : Member 1
Dataset: NASA C-MAPSS FD001
Purpose: Understand the dataset, visualise sensor patterns,
         calculate RUL, and identify useful sensors.

Run this file from the project root:
    python module1_eda/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────
# STEP 1 — DEFINE COLUMN NAMES
# ─────────────────────────────────────────
# NASA dataset has no headers — we add them manually
COLUMNS = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    's1',  's2',  's3',  's4',  's5',
    's6',  's7',  's8',  's9',  's10',
    's11', 's12', 's13', 's14', 's15',
    's16', 's17', 's18', 's19', 's20', 's21'
]

# ─────────────────────────────────────────
# STEP 2 — LOAD DATASET
# ─────────────────────────────────────────
print("=" * 60)
print("MODULE 1 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load training data
# sep='\s+' means columns are separated by spaces
# header=None means file has no header row
train_df = pd.read_csv(
    'data/train_FD001.txt',
    sep='\s+',
    header=None,
    names=COLUMNS
)

# Load test data
test_df = pd.read_csv(
    'data/test_FD001.txt',
    sep='\s+',
    header=None,
    names=COLUMNS
)

# Load RUL ground truth
rul_df = pd.read_csv(
    'data/RUL_FD001.txt',
    header=None,
    names=['RUL']
)

print("\n✅ Dataset loaded successfully!")

# ─────────────────────────────────────────
# STEP 3 — BASIC INFORMATION
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("BASIC INFORMATION")
print("─" * 60)
print(f"Training rows    : {train_df.shape[0]}")
print(f"Training columns : {train_df.shape[1]}")
print(f"Test rows        : {test_df.shape[0]}")
print(f"Number of engines in training : {train_df['engine_id'].nunique()}")
print(f"Number of engines in test     : {test_df['engine_id'].nunique()}")

# ─────────────────────────────────────────
# STEP 4 — FIRST LOOK AT DATA
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("FIRST 5 ROWS OF TRAINING DATA")
print("─" * 60)
print(train_df.head())

# ─────────────────────────────────────────
# STEP 5 — CHECK FOR MISSING VALUES
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("MISSING VALUES CHECK")
print("─" * 60)
missing = train_df.isnull().sum().sum()
print(f"Total missing values: {missing}")
if missing == 0:
    print("✅ No missing values — dataset is clean!")

# ─────────────────────────────────────────
# STEP 6 — ENGINE LIFE STATISTICS
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("ENGINE LIFE STATISTICS")
print("─" * 60)

# How many cycles did each engine run?
engine_life = train_df.groupby('engine_id')['cycle'].max()
print(f"Shortest engine life : {engine_life.min()} cycles")
print(f"Longest engine life  : {engine_life.max()} cycles")
print(f"Average engine life  : {engine_life.mean():.1f} cycles")

# ─────────────────────────────────────────
# STEP 7 — CALCULATE RUL
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("CALCULATING RUL")
print("─" * 60)

# RUL = max cycle for that engine - current cycle
# Get max cycle for each engine
max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.columns = ['engine_id', 'max_cycle']

# Merge back to main dataframe
train_df = train_df.merge(max_cycles, on='engine_id')

# Calculate RUL
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

# Drop the max_cycle column (no longer needed)
train_df.drop(columns=['max_cycle'], inplace=True)

print("✅ RUL column added successfully!")
print(f"\nSample RUL values for Engine 1:")
print(train_df[train_df['engine_id'] == 1][['engine_id', 'cycle', 'RUL']].head(10))

# ─────────────────────────────────────────
# STEP 8 — IDENTIFY CONSTANT SENSORS
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("IDENTIFYING CONSTANT SENSORS")
print("─" * 60)

sensor_cols = [f's{i}' for i in range(1, 22)]
constant_sensors = []
useful_sensors = []

for sensor in sensor_cols:
    std = train_df[sensor].std()
    if std < 0.001:  # Nearly zero standard deviation = constant
        constant_sensors.append(sensor)
        print(f"❌ {sensor} — CONSTANT (std={std:.6f}) → will be dropped")
    else:
        useful_sensors.append(sensor)

print(f"\n✅ Useful sensors  : {len(useful_sensors)} → {useful_sensors}")
print(f"❌ Constant sensors: {len(constant_sensors)} → {constant_sensors}")

# ─────────────────────────────────────────
# STEP 9 — SAVE OUTPUTS
# ─────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

# Save processed training data with RUL
train_df.to_csv('outputs/train_with_RUL.csv', index=False)
print("\n✅ Saved: outputs/train_with_RUL.csv")

# Save list of useful sensors for Module 2
import json
with open('outputs/useful_sensors.json', 'w') as f:
    json.dump(useful_sensors, f)
print("✅ Saved: outputs/useful_sensors.json")

# ─────────────────────────────────────────
# STEP 10 — VISUALISATIONS
# ─────────────────────────────────────────
print("\n" + "─" * 60)
print("GENERATING VISUALISATIONS")
print("─" * 60)

# ── Plot 1: Engine Life Distribution ──
plt.figure(figsize=(10, 5))
engine_life.hist(bins=30, color='steelblue', edgecolor='white')
plt.title('Engine Life Distribution\n(How many cycles did each engine run?)',
          fontsize=14)
plt.xlabel('Number of Cycles', fontsize=12)
plt.ylabel('Number of Engines', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plot1_engine_life_distribution.png', dpi=150)
plt.close()
print("✅ Saved: plot1_engine_life_distribution.png")

# ── Plot 2: RUL Distribution ──
plt.figure(figsize=(10, 5))
train_df['RUL'].hist(bins=50, color='coral', edgecolor='white')
plt.title('RUL Distribution\n(How many cycles remaining across all data points?)',
          fontsize=14)
plt.xlabel('Remaining Useful Life (cycles)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plot2_rul_distribution.png', dpi=150)
plt.close()
print("✅ Saved: plot2_rul_distribution.png")

# ── Plot 3: Sensor Degradation for One Engine ──
# Show how sensors change as engine approaches failure
engine1 = train_df[train_df['engine_id'] == 1]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Sensor Readings for Engine 1\n(Left = Healthy, Right = Failing)',
             fontsize=14)

# Pick 9 useful sensors to display
display_sensors = useful_sensors[:9]
for i, sensor in enumerate(display_sensors):
    row, col = i // 3, i % 3
    axes[row, col].plot(engine1['cycle'], engine1[sensor],
                        color='steelblue', linewidth=1)
    axes[row, col].set_title(sensor, fontsize=11)
    axes[row, col].set_xlabel('Cycle')
    axes[row, col].set_ylabel('Value')

plt.tight_layout()
plt.savefig('outputs/plot3_sensor_degradation.png', dpi=150)
plt.close()
print("✅ Saved: plot3_sensor_degradation.png")

# ── Plot 4: Correlation Heatmap of Useful Sensors ──
plt.figure(figsize=(12, 10))
corr_data = train_df[useful_sensors + ['RUL']].corr()
sns.heatmap(
    corr_data,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={'size': 8}
)
plt.title('Sensor Correlation Heatmap\n(How strongly each sensor correlates with RUL)',
          fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plot4_correlation_heatmap.png', dpi=150)
plt.close()
print("✅ Saved: plot4_correlation_heatmap.png")

# ── Plot 5: Average Sensor Trend (Healthy vs Degrading) ──
# Compare sensor values in early cycles vs late cycles
train_df['health_status'] = train_df['RUL'].apply(
    lambda x: 'Healthy (RUL > 100)' if x > 100 else 'Degrading (RUL ≤ 100)'
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Sensor Values: Healthy vs Degrading Engines',
             fontsize=14)

display_sensors2 = useful_sensors[:6]
for i, sensor in enumerate(display_sensors2):
    row, col = i // 3, i % 3
    for status, group in train_df.groupby('health_status'):
        axes[row, col].hist(
            group[sensor],
            bins=30,
            alpha=0.6,
            label=status,
            density=True
        )
    axes[row, col].set_title(sensor)
    axes[row, col].legend(fontsize=7)

plt.tight_layout()
plt.savefig('outputs/plot5_healthy_vs_degrading.png', dpi=150)
plt.close()
print("✅ Saved: plot5_healthy_vs_degrading.png")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("MODULE 1 COMPLETE — SUMMARY")
print("=" * 60)
print(f"✅ Dataset loaded           : {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"✅ Missing values           : None")
print(f"✅ RUL calculated           : Added as new column")
print(f"✅ Useful sensors identified: {len(useful_sensors)} sensors")
print(f"✅ Constant sensors dropped : {len(constant_sensors)} sensors")
print(f"✅ Plots generated          : 5 plots saved in outputs/")
print(f"✅ Files saved              : train_with_RUL.csv, useful_sensors.json")
print("\n👉 Next Step: Run Module 2 — Preprocessing")
print("=" * 60)