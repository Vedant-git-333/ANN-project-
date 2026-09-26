"""
============================================================
AeroPulse AI - Student 3
FILE: cnn_lstm_model.py

PURPOSE:
    Build, train, evaluate, and save a CNN-LSTM hybrid model
    for Remaining Useful Life (RUL) prediction on C-MAPSS FD001.

    The idea:
        - CNN layers extract LOCAL PATTERNS across consecutive
          cycles (e.g., sudden sensor spikes, short-term trends).
        - LSTM layers learn LONG-RANGE TEMPORAL DEPENDENCIES
          (how the extracted patterns change over time).
        - Combining both gives richer features than either alone.

    ARCHITECTURE:
        Input: (30, 17)   <- 30 cycles x 17 sensor features
            |
        Conv1D(64, kernel=3, relu)      <- extract local patterns
            |
        Conv1D(64, kernel=3, relu)      <- deeper local features
            |
        MaxPooling1D(pool=2)            <- downsample, reduce noise
            |
        LSTM(64)                        <- learn temporal evolution
            |
        Dropout(0.3)
            |
        Dense(32, relu)
            |
        Dense(1)                        <- predicted RUL (regression)

    REUSES:
        - data/sequences/*.npy   (from student3_prepare_sequences.py)
        - data/sequences/test_meta.csv

    PRODUCES:
        - models/cnn_lstm_model.keras
        - results/cnn_lstm_training_history.csv
        - results/cnn_lstm_predictions.csv
        - results/cnn_lstm_metrics.csv

HOW TO RUN (from inside Student 3 folder):
    python cnn_lstm_model.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

CNN_FILTERS_1   = 64      # Filters in first Conv1D layer
CNN_FILTERS_2   = 64      # Filters in second Conv1D layer
KERNEL_SIZE     = 3       # Convolution kernel/window size
POOL_SIZE       = 2       # MaxPooling reduction factor
LSTM_UNITS      = 64      # Units in the LSTM layer
DENSE_UNITS     = 32      # Units in the Dense layer
DROPOUT_RATE    = 0.3     # Dropout fraction
LEARNING_RATE   = 0.001   # Adam learning rate
BATCH_SIZE      = 128     # Samples per update
EPOCHS          = 100     # Max epochs (EarlyStopping controls actual)
PATIENCE        = 10      # EarlyStopping patience (same as ANN/CNN)


# ============================================================
# 3. PATHS
# ============================================================

SEQ_DIR     = "data/sequences"
MODELS_DIR  = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 4. LOAD SEQUENCE DATA
# ============================================================

print("Loading sequence data...")

X_train = np.load(os.path.join(SEQ_DIR, "X_train_seq.npy"))
y_train = np.load(os.path.join(SEQ_DIR, "y_train_seq.npy"))

X_val   = np.load(os.path.join(SEQ_DIR, "X_val_seq.npy"))
y_val   = np.load(os.path.join(SEQ_DIR, "y_val_seq.npy"))

X_test  = np.load(os.path.join(SEQ_DIR, "X_test_seq.npy"))
y_test  = np.load(os.path.join(SEQ_DIR, "y_test_seq.npy"))

# Test metadata for traceability
test_meta = pd.read_csv(os.path.join(SEQ_DIR, "test_meta.csv"))

print("Data loaded:")
print(f"  X_train : {X_train.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_test  : {y_test.shape}")

WINDOW_SIZE = X_train.shape[1]   # 30
N_FEATURES  = X_train.shape[2]   # 17

print(f"\nInput shape per sample: ({WINDOW_SIZE}, {N_FEATURES})")


# ============================================================
# 5. BUILD CNN-LSTM MODEL
# ============================================================

print("\n========== CNN-LSTM ARCHITECTURE ==========")

model = Sequential([

    # Input: (30 timesteps, 17 features)
    Input(shape=(WINDOW_SIZE, N_FEATURES)),

    # ---- CNN BLOCK ----
    # Conv1D slides a kernel across the time axis.
    # It reads CNN_FILTERS_1 local patterns of width KERNEL_SIZE=3 cycles.
    Conv1D(
        filters=CNN_FILTERS_1,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="same"      # keeps sequence length unchanged
    ),

    # Second Conv1D for deeper pattern extraction
    Conv1D(
        filters=CNN_FILTERS_2,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="same"
    ),

    # MaxPooling: reduces the sequence length by a factor of POOL_SIZE.
    # This compresses the CNN features before feeding to LSTM.
    MaxPooling1D(pool_size=POOL_SIZE),

    # ---- LSTM BLOCK ----
    # LSTM processes the CNN-extracted feature sequence.
    # return_sequences=False: only the final hidden state is used.
    LSTM(LSTM_UNITS, return_sequences=False),
    Dropout(DROPOUT_RATE),

    # ---- DENSE HEAD ----
    Dense(DENSE_UNITS, activation="relu"),

    # Regression output: single predicted RUL value
    Dense(1)

], name="CNN_LSTM_RUL")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 6. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, "cnn_lstm_best.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=0
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n========== TRAINING CNN-LSTM ==========")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[early_stopping, reduce_lr, checkpoint],

    verbose=1
)


# ============================================================
# 8. PREDICT ON TEST SET
# ============================================================

print("\n========== TESTING CNN-LSTM ==========")

y_pred = model.predict(X_test, verbose=0).flatten()


# ============================================================
# 9. EVALUATE
# ============================================================

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n========== CNN-LSTM RESULTS ==========")
print(f"  MAE  : {mae:.4f} cycles")
print(f"  RMSE : {rmse:.4f} cycles")
print(f"  R2   : {r2:.4f}")


# ============================================================
# 10. SAVE MODEL
# ============================================================

model.save(os.path.join(MODELS_DIR, "cnn_lstm_model.keras"))
print(f"\nSaved model: {MODELS_DIR}/cnn_lstm_model.keras")


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================
# Format matches Student 2: unit, cycle, actual_RUL, CNN_LSTM_prediction
# Student 4 merges on (unit, cycle, actual_RUL).

predictions = pd.DataFrame({
    "unit"               : test_meta["unit"].values,
    "cycle"              : test_meta["cycle"].values,
    "actual_RUL"         : y_test,
    "CNN_LSTM_prediction": y_pred,
})

predictions.to_csv(
    os.path.join(RESULTS_DIR, "cnn_lstm_predictions.csv"),
    index=False
)
print(f"Saved predictions: {RESULTS_DIR}/cnn_lstm_predictions.csv")


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": ["CNN_LSTM"],
    "MAE"  : [mae],
    "RMSE" : [rmse],
    "R2"   : [r2]
})

metrics.to_csv(
    os.path.join(RESULTS_DIR, "cnn_lstm_metrics.csv"),
    index=False
)
print(f"Saved metrics: {RESULTS_DIR}/cnn_lstm_metrics.csv")


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)
history_df.to_csv(
    os.path.join(RESULTS_DIR, "cnn_lstm_training_history.csv"),
    index=False
)
print(f"Saved history: {RESULTS_DIR}/cnn_lstm_training_history.csv")


# ============================================================
# 14. DONE
# ============================================================

print("\n================================")
print("CNN-LSTM TRAINING COMPLETE")
print("================================")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R2   = {r2:.4f}")
print(f"  Test samples: {len(y_pred):,}")
