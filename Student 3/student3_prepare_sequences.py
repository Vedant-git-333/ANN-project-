"""
============================================================
AeroPulse AI - Student 3
FILE: student3_prepare_sequences.py

PURPOSE:
    Build sliding-window sequences (30 cycles x 17 features)
    ready for LSTM and CNN-LSTM training.

    DATA SOURCE STRATEGY (in priority order):
    1. If Student 2's split files already exist under
       "../Student 2/data/splits/", load and use them directly.
    2. Otherwise, rebuild the same split from Student 1's
       raw file: "../Student 1/train_FD001.txt"
       using the EXACT same logic as student2_split.py.

    Either way:
        - The same 17 features as Student 2 are used.
        - The same 80/10/10 engine split is used.
        - StandardScaler is fit ONLY on the training set.
        - RUL is computed as (max_cycle - current_cycle),
          consistent with student2_preprocessing.py.
        - A window of 30 cycles is used (matching Student 1).

    Engine boundaries are STRICTLY respected when creating
    sequences - no cross-engine leakage.

PRODUCES:
    Student 3/data/sequences/
        X_train_seq.npy    (N_train, 30, 17)
        y_train_seq.npy    (N_train,)
        X_val_seq.npy      (N_val, 30, 17)
        y_val_seq.npy      (N_val,)
        X_test_seq.npy     (N_test, 30, 17)
        y_test_seq.npy     (N_test,)
        test_meta.csv      unit, cycle, actual_RUL per test sample

    Student 3/models/
        student3_scaler.pkl   <- StandardScaler (saved for reference)

HOW TO RUN:
    cd "Student 3"
    python student3_prepare_sequences.py
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from student3_sequence_utils import (
    WINDOW_SIZE,
    create_sequences,
    create_sequences_for_test,
    verify_sequence_shapes,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# The 17 features used by Student 2 (from Student 2/features.txt)
# Must be identical for a fair comparison across all 4 models.
FEATURES = [
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

TARGET = "RUL"

# Output directories (relative to Student 3 folder)
SEQ_DIR    = "data/sequences"
MODELS_DIR = "models"

os.makedirs(SEQ_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Paths to Student 2's pre-built split files (may not exist)
S2_TRAIN_PATH = "../Student 2/data/splits/train.csv"
S2_VAL_PATH   = "../Student 2/data/splits/validation.csv"
S2_TEST_PATH  = "../Student 2/data/splits/test.csv"
S2_SCALER     = "../Student 2/models/scaler.pkl"

# Path to Student 1's raw training file (always available)
S1_RAW_TRAIN  = "../Student 1/train_FD001.txt"


# ============================================================
# 2. LOAD OR BUILD SPLITS
# ============================================================

splits_exist = all(
    os.path.exists(p)
    for p in [S2_TRAIN_PATH, S2_VAL_PATH, S2_TEST_PATH]
)

if splits_exist:
    # ---- PATH A: Reuse Student 2's pre-built splits ----
    print("Found Student 2's split files. Loading them directly...")

    train_df = pd.read_csv(S2_TRAIN_PATH)
    val_df   = pd.read_csv(S2_VAL_PATH)
    test_df  = pd.read_csv(S2_TEST_PATH)

    print(f"  Train: {len(train_df):,} rows, {train_df['unit'].nunique()} engines")
    print(f"  Val  : {len(val_df):,} rows, {val_df['unit'].nunique()} engines")
    print(f"  Test : {len(test_df):,} rows, {test_df['unit'].nunique()} engines")

    # Load Student 2's scaler if it exists (no re-fitting)
    if os.path.exists(S2_SCALER):
        print("\nLoading Student 2's fitted scaler (no re-fitting)...")
        scaler = joblib.load(S2_SCALER)
        train_df = train_df.copy()
        val_df   = val_df.copy()
        test_df  = test_df.copy()
        train_df[FEATURES] = scaler.transform(train_df[FEATURES])
        val_df[FEATURES]   = scaler.transform(val_df[FEATURES])
        test_df[FEATURES]  = scaler.transform(test_df[FEATURES])
        print("  Scaler applied.")
    else:
        # Splits exist but scaler doesn't: fit our own scaler on train only
        print("\nStudent 2's scaler not found. Fitting a new StandardScaler on training data...")
        scaler = StandardScaler()
        train_df = train_df.copy()
        val_df   = val_df.copy()
        test_df  = test_df.copy()
        train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
        val_df[FEATURES]   = scaler.transform(val_df[FEATURES])
        test_df[FEATURES]  = scaler.transform(test_df[FEATURES])
        joblib.dump(scaler, os.path.join(MODELS_DIR, "student3_scaler.pkl"))
        print("  New scaler saved: models/student3_scaler.pkl")

else:
    # ---- PATH B: Rebuild splits from Student 1's raw file ----
    print("Student 2's split files not found.")
    print(f"Rebuilding from Student 1's raw file: {S1_RAW_TRAIN}")

    if not os.path.exists(S1_RAW_TRAIN):
        print(f"\nERROR: Cannot find {S1_RAW_TRAIN}")
        print("Please ensure Student 1's raw data file is present.")
        sys.exit(1)

    # --------------------------------------------------------
    # 2a. Load raw C-MAPSS FD001 data (same as student2_preprocessing.py)
    # --------------------------------------------------------
    columns = (
        ["unit", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(
        S1_RAW_TRAIN,
        sep=r"\s+",
        header=None,
        names=columns
    )

    print(f"  Loaded raw data: {df.shape}")

    # --------------------------------------------------------
    # 2b. Compute RUL (identical to student2_preprocessing.py)
    # --------------------------------------------------------
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    # --------------------------------------------------------
    # 2c. Drop constant/zero-variance sensors
    #     (same filtering Student 2 applied)
    # --------------------------------------------------------
    feature_columns = [
        col for col in df.columns
        if col not in ["unit", "cycle", "RUL"]
    ]

    constant_cols = [
        col for col in feature_columns
        if df[col].nunique() <= 1
    ]

    df = df.drop(columns=constant_cols)

    print(f"  Dropped constant columns: {constant_cols}")
    print(f"  Remaining columns: {df.columns.tolist()}")

    # --------------------------------------------------------
    # 2d. Engine-level split: 80/10/10  (same as student2_split.py)
    # --------------------------------------------------------
    engine_ids = sorted(df["unit"].unique())  # FD001: engines 1-100

    train_engines = engine_ids[:80]
    val_engines   = engine_ids[80:90]
    test_engines  = engine_ids[90:]

    train_df = df[df["unit"].isin(train_engines)].copy()
    val_df   = df[df["unit"].isin(val_engines)].copy()
    test_df  = df[df["unit"].isin(test_engines)].copy()

    print(
        f"\n  Split: {len(train_df):,} train rows ({len(train_engines)} engines) | "
        f"{len(val_df):,} val rows ({len(val_engines)} engines) | "
        f"{len(test_df):,} test rows ({len(test_engines)} engines)"
    )

    # --------------------------------------------------------
    # 2e. Normalize with StandardScaler (fit ONLY on train)
    # --------------------------------------------------------
    scaler = StandardScaler()

    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    val_df[FEATURES]   = scaler.transform(val_df[FEATURES])
    test_df[FEATURES]  = scaler.transform(test_df[FEATURES])

    # Save the scaler for reference
    joblib.dump(scaler, os.path.join(MODELS_DIR, "student3_scaler.pkl"))

    print("\n  StandardScaler fit on training data only.")
    print("  Saved: models/student3_scaler.pkl")


# ============================================================
# 3. BUILD SEQUENCES
# ============================================================

print(f"\nBuilding sequences (window size = {WINDOW_SIZE} cycles)...")

X_train, y_train, train_meta = create_sequences(
    train_df, WINDOW_SIZE, FEATURES, rul_col=TARGET
)

X_val, y_val, val_meta = create_sequences(
    val_df, WINDOW_SIZE, FEATURES, rul_col=TARGET
)

X_test, y_test, test_meta = create_sequences_for_test(
    test_df, WINDOW_SIZE, FEATURES, rul_col=TARGET
)

# Add actual_RUL to test metadata (needed for Student 4's handoff)
test_meta["actual_RUL"] = y_test


# ============================================================
# 4. VERIFY SHAPES
# ============================================================

verify_sequence_shapes(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    WINDOW_SIZE,
    len(FEATURES)
)


# ============================================================
# 5. SAVE TO DISK
# ============================================================

np.save(os.path.join(SEQ_DIR, "X_train_seq.npy"), X_train)
np.save(os.path.join(SEQ_DIR, "y_train_seq.npy"), y_train)

np.save(os.path.join(SEQ_DIR, "X_val_seq.npy"),   X_val)
np.save(os.path.join(SEQ_DIR, "y_val_seq.npy"),   y_val)

np.save(os.path.join(SEQ_DIR, "X_test_seq.npy"),  X_test)
np.save(os.path.join(SEQ_DIR, "y_test_seq.npy"),  y_test)

test_meta.to_csv(
    os.path.join(SEQ_DIR, "test_meta.csv"),
    index=False
)


# ============================================================
# 6. PRINT SUMMARY
# ============================================================

print("\n========================================")
print("SEQUENCE PREPARATION COMPLETE")
print("========================================")
print(f"Window size    : {WINDOW_SIZE} cycles")
print(f"Features       : {len(FEATURES)}")
print(f"Input shape    : ({WINDOW_SIZE}, {len(FEATURES)})")
print(f"Train samples  : {X_train.shape[0]:,}")
print(f"Val samples    : {X_val.shape[0]:,}")
print(f"Test samples   : {X_test.shape[0]:,}")
print(f"\nFiles saved to : {SEQ_DIR}/")
print("  X_train_seq.npy, y_train_seq.npy")
print("  X_val_seq.npy,   y_val_seq.npy")
print("  X_test_seq.npy,  y_test_seq.npy")
print("  test_meta.csv")
print("\nReady for: lstm_model.py  and  cnn_lstm_model.py")
