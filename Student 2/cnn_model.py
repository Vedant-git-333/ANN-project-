import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD DATA
# ============================================================

X_train = np.load("data/model_input/X_train.npy")
y_train = np.load("data/model_input/y_train.npy")

X_val = np.load("data/model_input/X_val.npy")
y_val = np.load("data/model_input/y_val.npy")

X_test = np.load("data/model_input/X_test.npy")
y_test = np.load("data/model_input/y_test.npy")


print("Original data:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# ============================================================
# 3. RESHAPE FOR CNN
# ============================================================

# Conv1D expects:
#
# (samples, timesteps/features, channels)
#
# Our data is:
#
# (samples, 17)
#
# Therefore:
#
# (samples, 17, 1)

X_train_cnn = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_val_cnn = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    1
)

X_test_cnn = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


print("\nCNN data:")
print("X_train:", X_train_cnn.shape)
print("X_val:  ", X_val_cnn.shape)
print("X_test: ", X_test_cnn.shape)


# ============================================================
# 4. BUILD CNN
# ============================================================

model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(1)
])


# ============================================================
# 5. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# 6. DISPLAY ARCHITECTURE
# ============================================================

print("\n========== CNN ARCHITECTURE ==========")

model.summary()


# ============================================================
# 7. CALLBACKS
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
# 8. TRAIN
# ============================================================

print("\n========== TRAINING CNN ==========")

history = model.fit(
    X_train_cnn,
    y_train,

    validation_data=(
        X_val_cnn,
        y_val
    ),

    epochs=100,
    batch_size=128,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 9. PREDICT
# ============================================================

print("\n========== TESTING CNN ==========")

y_pred = model.predict(
    X_test_cnn,
    verbose=0
).flatten()


# ============================================================
# 10. METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== CNN RESULTS ==========")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 11. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model.save(
    "models/cnn_model.keras"
)

print("\nSaved model:")
print("models/cnn_model.keras")


# ============================================================
# 12. LOAD TEST METADATA
# ============================================================

test_df = pd.read_csv(
    "data/splits/test.csv"
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "unit": test_df["unit"].values,
    "cycle": test_df["cycle"].values,
    "actual_RUL": y_test,
    "CNN_prediction": y_pred
})

os.makedirs("results", exist_ok=True)

predictions.to_csv(
    "results/cnn_predictions.csv",
    index=False
)

print("\nSaved predictions:")
print("results/cnn_predictions.csv")


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "model": ["CNN"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "results/cnn_metrics.csv",
    index=False
)

print("Saved metrics:")
print("results/cnn_metrics.csv")


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "results/cnn_training_history.csv",
    index=False
)

print("Saved training history:")
print("results/cnn_training_history.csv")


# ============================================================
# 16. FINISHED
# ============================================================

print("\n================================")
print("CNN TRAINING COMPLETE")
print("================================")
