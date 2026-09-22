import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

TRAIN_PATH = "data/splits/train.csv"
VAL_PATH = "data/splits/validation.csv"
TEST_PATH = "data/splits/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

# These are the features remaining after Student 2 preprocessing.
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
    "sensor_21"
]

TARGET = "RUL"


# ============================================================
# 3. CREATE X AND y
# ============================================================

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values

X_val = val_df[FEATURES].values
y_val = val_df[TARGET].values

X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values


print("\nOriginal X shapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# ============================================================
# 4. NORMALIZATION
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit ONLY on training data.
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler for validation and test.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


print("\nScaled X shapes:")
print("X_train:", X_train_scaled.shape)
print("X_val:  ", X_val_scaled.shape)
print("X_test: ", X_test_scaled.shape)


# ============================================================
# 5. CHECK NORMALIZATION
# ============================================================

print("\nTraining feature means after scaling:")
print(np.mean(X_train_scaled, axis=0))

print("\nTraining feature standard deviations:")
print(np.std(X_train_scaled, axis=0))


# ============================================================
# 6. CREATE DIRECTORIES
# ============================================================

os.makedirs("data/model_input", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 7. SAVE ARRAYS
# ============================================================

np.save("data/model_input/X_train.npy", X_train_scaled)
np.save("data/model_input/y_train.npy", y_train)

np.save("data/model_input/X_val.npy", X_val_scaled)
np.save("data/model_input/y_val.npy", y_val)

np.save("data/model_input/X_test.npy", X_test_scaled)
np.save("data/model_input/y_test.npy", y_test)


# ============================================================
# 8. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)


# ============================================================
# 9. SAVE FEATURE NAMES
# ============================================================

with open("data/model_input/features.txt", "w") as f:
    for feature in FEATURES:
        f.write(feature + "\n")


# ============================================================
# 10. FINISHED
# ============================================================

print("\n================================")
print("DATA PREPARATION COMPLETE")
print("================================")

print("\nSaved files:")

print("data/model_input/X_train.npy")
print("data/model_input/y_train.npy")

print("data/model_input/X_val.npy")
print("data/model_input/y_val.npy")

print("data/model_input/X_test.npy")
print("data/model_input/y_test.npy")

print("models/scaler.pkl")
print("data/model_input/features.txt")
