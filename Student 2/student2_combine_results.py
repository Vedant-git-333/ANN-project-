import pandas as pd
from pathlib import Path

# ============================================================
# STUDENT 2: COMBINE ANN + CNN RESULTS
# ============================================================

ANN_PRED_PATH = "results/ann_predictions.csv"
CNN_PRED_PATH = "results/cnn_predictions.csv"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

ann = pd.read_csv(ANN_PRED_PATH)
cnn = pd.read_csv(CNN_PRED_PATH)

print("ANN predictions:")
print(ann.head())
print("\nANN shape:", ann.shape)

print("\nCNN predictions:")
print(cnn.head())
print("\nCNN shape:", cnn.shape)


# ============================================================
# 2. CHECK COLUMN NAMES
# ============================================================

print("\nANN columns:", list(ann.columns))
print("CNN columns:", list(cnn.columns))


# ============================================================
# 3. MERGE ANN + CNN
# ============================================================

merged = pd.merge(
    ann,
    cnn,
    on=["unit", "cycle", "actual_RUL"],
    how="inner"
)

print("\nMerged shape:", merged.shape)


# ============================================================
# 4. CALCULATE INDIVIDUAL MODEL ERRORS
# ============================================================

merged["ANN_error"] = (
    merged["ANN_prediction"] - merged["actual_RUL"]
)

merged["CNN_error"] = (
    merged["CNN_prediction"] - merged["actual_RUL"]
)

merged["ANN_abs_error"] = merged["ANN_error"].abs()
merged["CNN_abs_error"] = merged["CNN_error"].abs()


# ============================================================
# 5. CALCULATE MODEL DISAGREEMENT
# ============================================================

merged["ANN_CNN_disagreement"] = (
    merged["ANN_prediction"] - merged["CNN_prediction"]
).abs()


# ============================================================
# 6. CALCULATE AVERAGE PREDICTION
# ============================================================

merged["ANN_CNN_average"] = (
    merged["ANN_prediction"] + merged["CNN_prediction"]
) / 2


# ============================================================
# 7. SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "unit",
    "cycle",
    "actual_RUL",
    "ANN_prediction",
    "CNN_prediction",
    "ANN_error",
    "CNN_error",
    "ANN_abs_error",
    "CNN_abs_error",
    "ANN_CNN_disagreement",
    "ANN_CNN_average"
]

merged = merged[final_columns]


# ============================================================
# 8. SAVE COMBINED PREDICTIONS
# ============================================================

output_path = OUTPUT_DIR / "student2_predictions.csv"

merged.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)


# ============================================================
# 9. MODEL COMPARISON TABLE
# ============================================================

ann_metrics = pd.read_csv("results/ann_metrics.csv")
cnn_metrics = pd.read_csv("results/cnn_metrics.csv")

comparison = pd.concat(
    [
        ann_metrics.assign(Model="ANN"),
        cnn_metrics.assign(Model="CNN")
    ],
    ignore_index=True
)

comparison_columns = ["Model"] + [
    col for col in comparison.columns if col != "Model"
]

comparison = comparison[comparison_columns]

comparison_path = OUTPUT_DIR / "student2_model_comparison.csv"

comparison.to_csv(comparison_path, index=False)

print("\nSaved:")
print(comparison_path)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n==========================================")
print("STUDENT 2 RESULTS SUMMARY")
print("==========================================")

print("\nModel comparison:")
print(comparison.to_string(index=False))

print("\nCombined prediction data:")
print(merged.head(10).to_string(index=False))

print("\nCombined shape:")
print(merged.shape)

print("\nDisagreement statistics:")
print(merged["ANN_CNN_disagreement"].describe())

print("\n==========================================")
print("STUDENT 2 COMBINATION COMPLETE")
print("==========================================")
