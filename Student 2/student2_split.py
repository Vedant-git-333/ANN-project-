import pandas as pd
import os

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

INPUT_PATH = "data/processed/fd001_processed.csv"

df = pd.read_csv(INPUT_PATH)

print("Total rows:", len(df))
print("Total engines:", df["unit"].nunique())


# ============================================================
# 2. GET ENGINE IDs
# ============================================================

engine_ids = sorted(df["unit"].unique())

print("\nEngine IDs:")
print(engine_ids)


# ============================================================
# 3. SPLIT BY ENGINE
# ============================================================

# FD001 has 100 engines.
#
# 80 engines -> training
# 10 engines -> validation
# 10 engines -> test

train_engines = engine_ids[:80]
val_engines = engine_ids[80:90]
test_engines = engine_ids[90:]


train_df = df[df["unit"].isin(train_engines)].copy()
val_df = df[df["unit"].isin(val_engines)].copy()
test_df = df[df["unit"].isin(test_engines)].copy()


# ============================================================
# 4. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n========== SPLIT ==========")

print(
    f"Training:   {len(train_df)} rows, "
    f"{train_df['unit'].nunique()} engines"
)

print(
    f"Validation: {len(val_df)} rows, "
    f"{val_df['unit'].nunique()} engines"
)

print(
    f"Test:       {len(test_df)} rows, "
    f"{test_df['unit'].nunique()} engines"
)


print("\nTraining engines:")
print(train_df["unit"].unique())

print("\nValidation engines:")
print(val_df["unit"].unique())

print("\nTest engines:")
print(test_df["unit"].unique())


# ============================================================
# 5. SAVE SPLITS
# ============================================================

os.makedirs("data/splits", exist_ok=True)

train_df.to_csv(
    "data/splits/train.csv",
    index=False
)

val_df.to_csv(
    "data/splits/validation.csv",
    index=False
)

test_df.to_csv(
    "data/splits/test.csv",
    index=False
)

print("\nSaved:")
print("data/splits/train.csv")
print("data/splits/validation.csv")
print("data/splits/test.csv")
