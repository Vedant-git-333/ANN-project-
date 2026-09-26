"""
============================================================
AeroPulse AI - Student 3
FILE: student3_evaluate.py

PURPOSE:
    Evaluate LSTM and CNN-LSTM on the test set and generate
    a combined model comparison table including ANN and CNN
    results from Student 2. Also generates plots.

    This script must be run AFTER:
        1. student3_prepare_sequences.py
        2. lstm_model.py
        3. cnn_lstm_model.py

PRODUCES:
    - results/student3_model_comparison.csv  <- all 4 models
    - results/student3_final_handoff.csv     <- for Student 4
    - results/plots/  <- training curves, scatter plots, etc.

HOW TO RUN:
    cd "Student 3"
    python student3_evaluate.py
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

RESULTS_DIR   = Path("results")
PLOTS_DIR     = RESULTS_DIR / "plots"
STUDENT2_DIR  = Path("../Student 2")

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD STUDENT 2 METRICS (ANN, CNN)
# ============================================================

print("Loading Student 2 metrics...")

ann_metrics = pd.read_csv(STUDENT2_DIR / "ann_metrics.csv")
cnn_metrics = pd.read_csv(STUDENT2_DIR / "cnn_metrics.csv")

# Extract scalar values
ann_mae  = float(ann_metrics["MAE"].iloc[0])
ann_rmse = float(ann_metrics["RMSE"].iloc[0])
ann_r2   = float(ann_metrics["R2"].iloc[0])

cnn_mae  = float(cnn_metrics["MAE"].iloc[0])
cnn_rmse = float(cnn_metrics["RMSE"].iloc[0])
cnn_r2   = float(cnn_metrics["R2"].iloc[0])


# ============================================================
# 3. LOAD STUDENT 3 METRICS (LSTM, CNN-LSTM)
# ============================================================

print("Loading Student 3 metrics...")

lstm_metrics     = pd.read_csv(RESULTS_DIR / "lstm_metrics.csv")
cnn_lstm_metrics = pd.read_csv(RESULTS_DIR / "cnn_lstm_metrics.csv")

lstm_mae      = float(lstm_metrics["MAE"].iloc[0])
lstm_rmse     = float(lstm_metrics["RMSE"].iloc[0])
lstm_r2       = float(lstm_metrics["R2"].iloc[0])

cnn_lstm_mae  = float(cnn_lstm_metrics["MAE"].iloc[0])
cnn_lstm_rmse = float(cnn_lstm_metrics["RMSE"].iloc[0])
cnn_lstm_r2   = float(cnn_lstm_metrics["R2"].iloc[0])


# ============================================================
# 4. COMBINED MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": ["ANN", "CNN", "LSTM", "CNN_LSTM"],
    "MAE"  : [ann_mae,  cnn_mae,  lstm_mae,  cnn_lstm_mae],
    "RMSE" : [ann_rmse, cnn_rmse, lstm_rmse, cnn_lstm_rmse],
    "R2"   : [ann_r2,   cnn_r2,   lstm_r2,   cnn_lstm_r2],
})

print("\n========== ALL 4 MODELS - TEST PERFORMANCE ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS_DIR / "student3_model_comparison.csv",
    index=False
)
print("\nSaved: results/student3_model_comparison.csv")


# ============================================================
# 5. LOAD PREDICTIONS FOR PLOTTING
# ============================================================

lstm_preds     = pd.read_csv(RESULTS_DIR / "lstm_predictions.csv")
cnn_lstm_preds = pd.read_csv(RESULTS_DIR / "cnn_lstm_predictions.csv")

ann_preds = pd.read_csv(STUDENT2_DIR / "ann_predictions.csv")
cnn_preds = pd.read_csv(STUDENT2_DIR / "cnn_predictions.csv")

# Load training histories for loss curves
lstm_history     = pd.read_csv(RESULTS_DIR / "lstm_training_history.csv")
cnn_lstm_history = pd.read_csv(RESULTS_DIR / "cnn_lstm_training_history.csv")


# ============================================================
# 6. PLOT: LSTM TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(lstm_history["loss"],     label="Training Loss")
plt.plot(lstm_history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM: Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_lstm_training_curve.png", dpi=200)
plt.close()
print("Saved: plots/01_lstm_training_curve.png")


# ============================================================
# 7. PLOT: CNN-LSTM TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(cnn_lstm_history["loss"],     label="Training Loss")
plt.plot(cnn_lstm_history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CNN-LSTM: Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_cnn_lstm_training_curve.png", dpi=200)
plt.close()
print("Saved: plots/02_cnn_lstm_training_curve.png")


# ============================================================
# 8. PLOT: LSTM - ACTUAL VS PREDICTED
# ============================================================

actual = lstm_preds["actual_RUL"]
pred   = lstm_preds["LSTM_prediction"]

plt.figure(figsize=(7, 7))
plt.scatter(actual, pred, alpha=0.35, s=12)
min_v = min(actual.min(), pred.min())
max_v = max(actual.max(), pred.max())
plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Perfect prediction")
plt.xlabel("Actual RUL (cycles)")
plt.ylabel("LSTM Predicted RUL (cycles)")
plt.title("LSTM: Actual vs Predicted RUL")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_lstm_actual_vs_predicted.png", dpi=200)
plt.close()
print("Saved: plots/03_lstm_actual_vs_predicted.png")


# ============================================================
# 9. PLOT: CNN-LSTM - ACTUAL VS PREDICTED
# ============================================================

actual = cnn_lstm_preds["actual_RUL"]
pred   = cnn_lstm_preds["CNN_LSTM_prediction"]

plt.figure(figsize=(7, 7))
plt.scatter(actual, pred, alpha=0.35, s=12, color="orange")
min_v = min(actual.min(), pred.min())
max_v = max(actual.max(), pred.max())
plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Perfect prediction")
plt.xlabel("Actual RUL (cycles)")
plt.ylabel("CNN-LSTM Predicted RUL (cycles)")
plt.title("CNN-LSTM: Actual vs Predicted RUL")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_cnn_lstm_actual_vs_predicted.png", dpi=200)
plt.close()
print("Saved: plots/04_cnn_lstm_actual_vs_predicted.png")


# ============================================================
# 10. PLOT: PREDICTION ERROR DISTRIBUTION (LSTM vs CNN-LSTM)
# ============================================================

lstm_error     = lstm_preds["LSTM_prediction"]     - lstm_preds["actual_RUL"]
cnn_lstm_error = cnn_lstm_preds["CNN_LSTM_prediction"] - cnn_lstm_preds["actual_RUL"]

plt.figure(figsize=(9, 5))
plt.hist(lstm_error,     bins=40, alpha=0.6, label="LSTM")
plt.hist(cnn_lstm_error, bins=40, alpha=0.6, label="CNN-LSTM")
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("LSTM vs CNN-LSTM: Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_lstm_vs_cnn_lstm_error_dist.png", dpi=200)
plt.close()
print("Saved: plots/05_lstm_vs_cnn_lstm_error_dist.png")


# ============================================================
# 11. PLOT: ALL 4 MODELS - MAE / RMSE / R2 COMPARISON BAR CHARTS
# ============================================================

models = comparison["Model"].tolist()
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# MAE
plt.figure(figsize=(8, 5))
bars = plt.bar(models, comparison["MAE"], color=colors)
for bar, val in zip(bars, comparison["MAE"]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5, f"{val:.2f}", ha="center", fontsize=9)
plt.ylabel("MAE (cycles)")
plt.title("All 4 Models: Mean Absolute Error Comparison")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_all_models_mae.png", dpi=200)
plt.close()
print("Saved: plots/06_all_models_mae.png")

# RMSE
plt.figure(figsize=(8, 5))
bars = plt.bar(models, comparison["RMSE"], color=colors)
for bar, val in zip(bars, comparison["RMSE"]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5, f"{val:.2f}", ha="center", fontsize=9)
plt.ylabel("RMSE (cycles)")
plt.title("All 4 Models: Root Mean Squared Error Comparison")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_all_models_rmse.png", dpi=200)
plt.close()
print("Saved: plots/07_all_models_rmse.png")

# R2
plt.figure(figsize=(8, 5))
bars = plt.bar(models, comparison["R2"], color=colors)
for bar, val in zip(bars, comparison["R2"]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
plt.ylabel("R\u00b2 Score")
plt.title("All 4 Models: R\u00b2 Comparison")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_all_models_r2.png", dpi=200)
plt.close()
print("Saved: plots/08_all_models_r2.png")


# ============================================================
# 12. PLOT: EXAMPLE ENGINE DEGRADATION CURVE
# ============================================================
# Pick the first engine in the test set and plot its full
# degradation trajectory with LSTM and CNN-LSTM overlaid.

example_unit = lstm_preds["unit"].iloc[0]

lstm_eng     = lstm_preds[lstm_preds["unit"] == example_unit].copy()
cnn_lstm_eng = cnn_lstm_preds[cnn_lstm_preds["unit"] == example_unit].copy()

plt.figure(figsize=(10, 5))
plt.plot(lstm_eng["cycle"],     lstm_eng["actual_RUL"],
         "k-",  lw=2,   label="Actual RUL")
plt.plot(lstm_eng["cycle"],     lstm_eng["LSTM_prediction"],
         "b--", lw=1.5, label="LSTM Prediction")
plt.plot(cnn_lstm_eng["cycle"], cnn_lstm_eng["CNN_LSTM_prediction"],
         "r--", lw=1.5, label="CNN-LSTM Prediction")
plt.xlabel("Cycle")
plt.ylabel("RUL (cycles)")
plt.title(f"Engine {example_unit}: RUL Degradation Curve (LSTM vs CNN-LSTM)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_example_engine_degradation.png", dpi=200)
plt.close()
print("Saved: plots/09_example_engine_degradation.png")


# ============================================================
# 13. STUDENT 4 FINAL HANDOFF FILE
# ============================================================
# Merge LSTM and CNN-LSTM predictions on (unit, cycle, actual_RUL).
# Student 4 will further merge this with Student 2's ANN/CNN.

handoff = pd.merge(
    lstm_preds,
    cnn_lstm_preds[["unit", "cycle", "actual_RUL", "CNN_LSTM_prediction"]],
    on=["unit", "cycle", "actual_RUL"],
    how="inner"
)

handoff.to_csv(
    RESULTS_DIR / "student3_final_handoff.csv",
    index=False
)

print(f"\nSaved Student 4 handoff: results/student3_final_handoff.csv")
print(f"  Rows: {len(handoff)}")
print(f"  Columns: {list(handoff.columns)}")


# ============================================================
# 14. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("AEROPULSE AI - STUDENT 3 EVALUATION COMPLETE")
print("=" * 60)

print("\n--- ALL 4 MODELS: TEST SET PERFORMANCE ---")
for _, row in comparison.iterrows():
    print(
        f"  {row['Model']:<10}  MAE={row['MAE']:.4f}  "
        f"RMSE={row['RMSE']:.4f}  R2={row['R2']:.4f}"
    )

print("\n--- OUTPUT FILES ---")
print("  results/student3_model_comparison.csv")
print("  results/lstm_predictions.csv")
print("  results/cnn_lstm_predictions.csv")
print("  results/lstm_metrics.csv")
print("  results/cnn_lstm_metrics.csv")
print("  results/student3_final_handoff.csv")
print("  results/plots/  (9 plots)")
print("\n  Ready for Student 4.")
