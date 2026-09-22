import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD RAW C-MAPSS FD001 DATA
# ============================================================

DATA_PATH = "data/train_FD001.txt"

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CALCULATE RUL
# ============================================================

# Find the maximum cycle reached by each engine
max_cycle = df.groupby("unit")["cycle"].transform("max")

# RUL = final cycle of engine - current cycle
df["RUL"] = max_cycle - df["cycle"]

print("\nDataset with RUL:")
print(df.head())

print("\nRUL statistics:")
print(df["RUL"].describe())


# ============================================================
# 3. REMOVE CONSTANT COLUMNS
# ============================================================

# Columns that contain only one unique value provide no
# information to a neural network.
feature_columns = [
    col for col in df.columns
    if col not in ["unit", "cycle", "RUL"]
]

constant_columns = [
    col for col in feature_columns
    if df[col].nunique() <= 1
]

print("\nConstant feature columns:")
print(constant_columns)

df = df.drop(columns=constant_columns)

feature_columns = [
    col for col in df.columns
    if col not in ["unit", "cycle", "RUL"]
]

print("\nRemaining features:")
print(feature_columns)


# ============================================================
# 4. SAVE PROCESSED DATA
# ============================================================

import os

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/fd001_processed.csv"

df.to_csv(output_path, index=False)

print("\nSaved processed dataset to:")
print(output_path)

print("\nFinal shape:", df.shape)
print("\nFinal columns:")
print(df.columns.tolist())