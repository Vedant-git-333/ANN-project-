"""
============================================================
AeroPulse AI - Student 3
FILE: student3_smoke_test.py

PURPOSE:
    Quick pipeline smoke test. Validates:

    1. Student 1's raw data can be loaded
    2. RUL computation and feature selection are correct
    3. 80/10/10 engine split works
    4. StandardScaler scales correctly (fit on train only)
    5. Sequence generation produces correct 3D shapes
    6. No engine-boundary leakage in sequences
    7. LSTM model builds + forward pass works
    8. CNN-LSTM model builds + forward pass works
    9. Prediction shapes align with metadata

    Uses only 2 mini-epochs. Does NOT do full training.
    Runs in < 60 seconds to give quick confidence.

HOW TO RUN:
    cd "Student 3"
    python student3_smoke_test.py
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from student3_sequence_utils import (
    WINDOW_SIZE,
    create_sequences,
    create_sequences_for_test,
    verify_sequence_shapes,
)


print("=" * 60)
print("STUDENT 3 SMOKE TEST")
print("=" * 60)


# ============================================================
# STEP 1: Find and load raw data
# ============================================================

print("\n[1] Locating raw data...")

RAW_PATH = "../Student 1/train_FD001.txt"
if not os.path.exists(RAW_PATH):
    print(f"  ERROR: {RAW_PATH} not found.")
    sys.exit(1)

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df = pd.read_csv(RAW_PATH, sep=r"\s+", header=None, names=columns)

# Compute RUL
max_cycle = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]

# Drop constant columns
feature_cols = [c for c in df.columns if c not in ["unit", "cycle", "RUL"]]
constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
df = df.drop(columns=constant_cols)

print(f"  Loaded: {df.shape[0]:,} rows, {df['unit'].nunique()} engines")
print(f"  Dropped constant columns: {constant_cols}")
print("  [OK] Raw data loaded.")


# ============================================================
# STEP 2: Split 80/10/10
# ============================================================

print("\n[2] Applying 80/10/10 engine split...")

engine_ids    = sorted(df["unit"].unique())
train_engines = engine_ids[:80]
val_engines   = engine_ids[80:90]
test_engines  = engine_ids[90:]

train_df = df[df["unit"].isin(train_engines)].copy()
val_df   = df[df["unit"].isin(val_engines)].copy()
test_df  = df[df["unit"].isin(test_engines)].copy()

print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows")
assert train_df["unit"].nunique() == 80
assert val_df["unit"].nunique()   == 10
assert test_df["unit"].nunique()  == 10
print("  [OK] Split verified.")


# ============================================================
# STEP 3: Scale features
# ============================================================

print("\n[3] Applying StandardScaler (fit on train only)...")

FEATURES = [
    "setting_1", "setting_2",
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_6",
    "sensor_7",  "sensor_8",  "sensor_9",  "sensor_11",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]

# Confirm all features are present after dropping constants
missing = [f for f in FEATURES if f not in train_df.columns]
if missing:
    print(f"  ERROR: Features missing from data: {missing}")
    sys.exit(1)

scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
val_df[FEATURES]   = scaler.transform(val_df[FEATURES])
test_df[FEATURES]  = scaler.transform(test_df[FEATURES])

train_mean = train_df[FEATURES].mean().round(4)
assert all(abs(train_mean) < 0.01), "Train feature means should be ~0 after scaling"
print(f"  Train feature mean (check ~0): {train_mean.values[:3]} ...")
print("  [OK] Scaling verified.")


# ============================================================
# STEP 4: Build sequences
# ============================================================

print(f"\n[4] Building sequences (window={WINDOW_SIZE})...")

X_train, y_train, _ = create_sequences(train_df, WINDOW_SIZE, FEATURES)
X_val,   y_val,   _ = create_sequences(val_df,   WINDOW_SIZE, FEATURES)
X_test,  y_test,  test_meta = create_sequences_for_test(test_df, WINDOW_SIZE, FEATURES)

verify_sequence_shapes(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    WINDOW_SIZE, len(FEATURES)
)
print("  [OK] Sequence shapes verified.")


# ============================================================
# STEP 5: Engine boundary check
# ============================================================

print("\n[5] Checking engine boundary integrity...")

test_units_expected = set(test_engines)
test_units_in_meta  = set(test_meta["unit"].unique())
unexpected = test_units_in_meta - test_units_expected

if unexpected:
    print(f"  ERROR: Sequences contain unexpected engine IDs: {unexpected}")
    sys.exit(1)

# Confirm no test engine data leaked into training sequences
train_units_in_meta = set(create_sequences(train_df, WINDOW_SIZE, FEATURES)[2]["unit"].unique())
overlap = train_units_in_meta & set(test_engines)
if overlap:
    print(f"  ERROR: Test engines found in training sequences: {overlap}")
    sys.exit(1)

print(f"  Test engines in sequences: {sorted(test_units_in_meta)}")
print("  [OK] No cross-boundary leakage detected.")


# ============================================================
# STEP 6: LSTM smoke run
# ============================================================

print("\n[6] LSTM smoke run (2 epochs, 500 samples)...")

np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D

lstm_model = Sequential([
    Input(shape=(WINDOW_SIZE, len(FEATURES))),
    LSTM(16, return_sequences=True),
    Dropout(0.2),
    LSTM(8, return_sequences=False),
    Dense(1)
], name="LSTM_smoke")

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.fit(
    X_train[:500], y_train[:500],
    validation_data=(X_val[:100], y_val[:100]),
    epochs=2, batch_size=64, verbose=0
)

sample_pred = lstm_model.predict(X_test[:5], verbose=0).flatten()
assert sample_pred.shape == (5,), f"Wrong shape: {sample_pred.shape}"
print(f"  Sample LSTM predictions: {sample_pred.round(2)}")
print("  [OK] LSTM smoke run passed.")


# ============================================================
# STEP 7: CNN-LSTM smoke run
# ============================================================

print("\n[7] CNN-LSTM smoke run (2 epochs, 500 samples)...")

cnn_lstm_model = Sequential([
    Input(shape=(WINDOW_SIZE, len(FEATURES))),
    Conv1D(16, kernel_size=3, activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),
    LSTM(16, return_sequences=False),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1)
], name="CNN_LSTM_smoke")

cnn_lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
cnn_lstm_model.fit(
    X_train[:500], y_train[:500],
    validation_data=(X_val[:100], y_val[:100]),
    epochs=2, batch_size=64, verbose=0
)

sample_pred = cnn_lstm_model.predict(X_test[:5], verbose=0).flatten()
assert sample_pred.shape == (5,), f"Wrong shape: {sample_pred.shape}"
print(f"  Sample CNN-LSTM predictions: {sample_pred.round(2)}")
print("  [OK] CNN-LSTM smoke run passed.")


# ============================================================
# STEP 8: Prediction-metadata alignment
# ============================================================

print("\n[8] Checking prediction-metadata alignment...")

full_lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
assert full_lstm_pred.shape[0] == len(test_meta), (
    f"LSTM predictions ({full_lstm_pred.shape[0]}) != metadata rows ({len(test_meta)})"
)
assert full_lstm_pred.shape[0] == len(y_test), (
    f"LSTM predictions ({full_lstm_pred.shape[0]}) != y_test ({len(y_test)})"
)
print(f"  Predictions: {full_lstm_pred.shape[0]:,} | Metadata: {len(test_meta):,} | Labels: {len(y_test):,}")
print("  [OK] Alignment confirmed.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
print(f"\n  Window size      : {WINDOW_SIZE}")
print(f"  Features         : {len(FEATURES)}")
print(f"  Train sequences  : {X_train.shape[0]:,}")
print(f"  Val sequences    : {X_val.shape[0]:,}")
print(f"  Test sequences   : {X_test.shape[0]:,}")
print(f"  Input shape      : {X_train.shape[1:]}")
print(f"\nRun full training with:")
print("  python student3_prepare_sequences.py")
print("  python lstm_model.py")
print("  python cnn_lstm_model.py")
print("  python student3_evaluate.py")
