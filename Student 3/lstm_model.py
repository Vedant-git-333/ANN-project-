"""
============================================================
AeroPulse AI - Student 3
FILE: lstm_model.py

PURPOSE:
    Build, train, evaluate, and save an LSTM model for
    Remaining Useful Life (RUL) prediction on C-MAPSS FD001.

    ARCHITECTURE:
        Input: (30, 17)  <- 30 cycles x 17 sensor features
            |
        LSTM(64, return_sequences=True)
            |
        Dropout(0.2)
            |
        LSTM(32)
            |
        Dropout(0.2)
            |
        Dense(16, relu)
            |
        Dense(1)           <- predicted RUL (regression)

    REUSES:
        - data/sequences/*.npy   (from student3_prepare_sequences.py)
        - data/sequences/test_meta.csv

    PRODUCES:
        - models/lstm_model.keras
        - results/lstm_training_history.csv
        - results/lstm_predictions.csv
        - results/lstm_metrics.csv

HOW TO RUN (from inside Student 3 folder):
    python lstm_model.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
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
# Adjust these values to experiment with different configurations.

LSTM_UNITS_1    = 64      # Units in the first LSTM layer
LSTM_UNITS_2    = 32      # Units in the second LSTM layer
DENSE_UNITS     = 16      # Units in the intermediate Dense layer
DROPOUT_RATE    = 0.2     # Dropout fraction (prevents overfitting)
LEARNING_RATE   = 0.001   # Adam optimizer learning rate
BATCH_SIZE      = 128     # Samples per gradient update
EPOCHS          = 100     # Maximum training epochs (EarlyStopping controls actual)
PATIENCE        = 10      # EarlyStopping patience (same as Student 2's ANN/CNN)


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

# Test metadata for traceability (unit, cycle, actual_RUL)
test_meta = pd.read_csv(os.path.join(SEQ_DIR, "test_meta.csv"))

print("Data loaded:")
print(f"  X_train : {X_train.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_test  : {y_test.shape}")

# Determine input dimensions from loaded data
WINDOW_SIZE = X_train.shape[1]   # number of timesteps (= 30)
N_FEATURES  = X_train.shape[2]   # number of features  (= 17)

print(f"\nInput shape per sample: ({WINDOW_SIZE}, {N_FEATURES})")


# ============================================================
# 5. BUILD LSTM MODEL
# ============================================================

print("\n========== LSTM ARCHITECTURE ==========")

model = Sequential([

    # Input layer: explicitly declares expected shape
    Input(shape=(WINDOW_SIZE, N_FEATURES)),

    # First LSTM: return_sequences=True because a second LSTM follows
    LSTM(LSTM_UNITS_1, return_sequences=True),
    Dropout(DROPOUT_RATE),

    # Second LSTM: return_sequences=False (only last output goes forward)
    LSTM(LSTM_UNITS_2, return_sequences=False),
    Dropout(DROPOUT_RATE),

    # Intermediate fully-connected layer
    Dense(DENSE_UNITS, activation="relu"),

    # Output: single neuron, no activation = regression output
    Dense(1)

], name="LSTM_RUL")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",     # MSE loss for regression (same as ANN/CNN)
    metrics=["mae"] # Track MAE during training for monitoring
)

model.summary()


# ============================================================
# 6. CALLBACKS
# ============================================================
# These are the same callback types used by Student 2's ANN/CNN.

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,  # revert to best epoch when stopping
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
    filepath=os.path.join(MODELS_DIR, "lstm_best.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=0
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n========== TRAINING LSTM ==========")

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

print("\n========== TESTING LSTM ==========")

y_pred = model.predict(X_test, verbose=0).flatten()


# ============================================================
# 9. EVALUATE
# ============================================================

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n========== LSTM RESULTS ==========")
print(f"  MAE  : {mae:.4f} cycles")
print(f"  RMSE : {rmse:.4f} cycles")
print(f"  R2   : {r2:.4f}")


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
print(f"\nSaved model: {MODELS_DIR}/lstm_model.keras")


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================
# Format matches Student 2 exactly: unit, cycle, actual_RUL, LSTM_prediction
# Student 4 will merge this with ANN/CNN predictions on (unit, cycle, actual_RUL).

predictions = pd.DataFrame({
    "unit"            : test_meta["unit"].values,
    "cycle"           : test_meta["cycle"].values,
    "actual_RUL"      : y_test,
    "LSTM_prediction" : y_pred,
})

predictions.to_csv(
    os.path.join(RESULTS_DIR, "lstm_predictions.csv"),
    index=False
)
print(f"Saved predictions: {RESULTS_DIR}/lstm_predictions.csv")


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": ["LSTM"],
    "MAE"  : [mae],
    "RMSE" : [rmse],
    "R2"   : [r2]
})

metrics.to_csv(
    os.path.join(RESULTS_DIR, "lstm_metrics.csv"),
    index=False
)
print(f"Saved metrics: {RESULTS_DIR}/lstm_metrics.csv")


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)
history_df.to_csv(
    os.path.join(RESULTS_DIR, "lstm_training_history.csv"),
    index=False
)
print(f"Saved training history: {RESULTS_DIR}/lstm_training_history.csv")


# ============================================================
# 14. DONE
# ============================================================

print("\n================================")
print("LSTM TRAINING COMPLETE")
print("================================")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R2   = {r2:.4f}")
print(f"  Test samples: {len(y_pred):,}")
