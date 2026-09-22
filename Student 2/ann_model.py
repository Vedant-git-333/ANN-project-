import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

X_train = np.load("data/model_input/X_train.npy")
y_train = np.load("data/model_input/y_train.npy")

X_val = np.load("data/model_input/X_val.npy")
y_val = np.load("data/model_input/y_val.npy")

X_test = np.load("data/model_input/X_test.npy")
y_test = np.load("data/model_input/y_test.npy")


print("Data loaded:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:  ", X_val.shape)
print("y_val:  ", y_val.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# ============================================================
# 3. BUILD ANN
# ============================================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(128, activation="relu"),
    Dropout(0.2),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])


# ============================================================
# 4. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# 5. DISPLAY MODEL
# ============================================================

print("\n========== ANN ARCHITECTURE ==========")

model.summary()


# ============================================================
# 6. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n========== TRAINING ANN ==========")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=100,
    batch_size=128,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 8. PREDICT ON TEST SET
# ============================================================

print("\n========== TESTING ANN ==========")

y_pred = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========== ANN RESULTS ==========")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 10. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save(
    "models/ann_model.keras"
)

print("\nSaved model:")
print("models/ann_model.keras")


# ============================================================
# 11. LOAD TEST METADATA
# ============================================================

test_df = pd.read_csv(
    "data/splits/test.csv"
)


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "unit": test_df["unit"].values,
    "cycle": test_df["cycle"].values,
    "actual_RUL": y_test,
    "ANN_prediction": y_pred
})

os.makedirs("results", exist_ok=True)

predictions.to_csv(
    "results/ann_predictions.csv",
    index=False
)

print("\nSaved predictions:")
print("results/ann_predictions.csv")


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": ["ANN"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "results/ann_metrics.csv",
    index=False
)

print("Saved metrics:")
print("results/ann_metrics.csv")


# ============================================================
# 14. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "results/ann_training_history.csv",
    index=False
)

print("Saved training history:")
print("results/ann_training_history.csv")


print("\n================================")
print("ANN TRAINING COMPLETE")
print("================================")
