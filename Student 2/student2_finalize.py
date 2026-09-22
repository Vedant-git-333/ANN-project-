import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# AEROPULSE AI
# STUDENT 2 FINALIZATION
# ANN + CNN ANALYSIS
# ============================================================

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

predictions = pd.read_csv(
    RESULTS_DIR / "student2_predictions.csv"
)

ann_metrics = pd.read_csv(
    RESULTS_DIR / "ann_metrics.csv"
)

cnn_metrics = pd.read_csv(
    RESULTS_DIR / "cnn_metrics.csv"
)

ann_history = pd.read_csv(
    RESULTS_DIR / "ann_training_history.csv"
)

cnn_history = pd.read_csv(
    RESULTS_DIR / "cnn_training_history.csv"
)


print("Loaded Student 2 results.")
print("Prediction shape:", predictions.shape)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

required_columns = [
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

missing_columns = [
    col for col in required_columns
    if col not in predictions.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if predictions.shape[0] != 2251:
    print(
        f"WARNING: Expected 2251 test observations, "
        f"found {predictions.shape[0]}"
    )

if predictions[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in prediction results."
    )

print("Validation checks passed.")


# ============================================================
# 3. FINAL HANDOFF FILE
# ============================================================

handoff_columns = [
    "unit",
    "cycle",
    "actual_RUL",
    "ANN_prediction",
    "CNN_prediction",
    "ANN_abs_error",
    "CNN_abs_error",
    "ANN_CNN_disagreement",
    "ANN_CNN_average"
]

handoff = predictions[handoff_columns].copy()

handoff_path = RESULTS_DIR / "student2_final_handoff.csv"

handoff.to_csv(
    handoff_path,
    index=False
)

print("\nSaved final handoff:")
print(handoff_path)


# ============================================================
# 4. MODEL COMPARISON PLOT
# ============================================================

comparison = pd.DataFrame({
    "Model": ["ANN", "CNN"],
    "MAE": [
        float(ann_metrics["MAE"].iloc[0]),
        float(cnn_metrics["MAE"].iloc[0])
    ],
    "RMSE": [
        float(ann_metrics["RMSE"].iloc[0]),
        float(cnn_metrics["RMSE"].iloc[0])
    ],
    "R2": [
        float(ann_metrics["R2"].iloc[0]),
        float(cnn_metrics["R2"].iloc[0])
    ]
})

comparison.to_csv(
    RESULTS_DIR / "student2_model_comparison_clean.csv",
    index=False
)


# MAE
plt.figure(figsize=(7, 5))
plt.bar(comparison["Model"], comparison["MAE"])
plt.ylabel("MAE (cycles)")
plt.title("ANN vs CNN - Mean Absolute Error")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "01_model_mae_comparison.png",
    dpi=200
)
plt.close()


# RMSE
plt.figure(figsize=(7, 5))
plt.bar(comparison["Model"], comparison["RMSE"])
plt.ylabel("RMSE (cycles)")
plt.title("ANN vs CNN - Root Mean Squared Error")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "02_model_rmse_comparison.png",
    dpi=200
)
plt.close()


# R2
plt.figure(figsize=(7, 5))
plt.bar(comparison["Model"], comparison["R2"])
plt.ylabel("R²")
plt.title("ANN vs CNN - R²")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "03_model_r2_comparison.png",
    dpi=200
)
plt.close()


# ============================================================
# 5. ACTUAL VS PREDICTED - ANN
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    predictions["actual_RUL"],
    predictions["ANN_prediction"],
    alpha=0.35,
    s=15
)

min_value = min(
    predictions["actual_RUL"].min(),
    predictions["ANN_prediction"].min()
)

max_value = max(
    predictions["actual_RUL"].max(),
    predictions["ANN_prediction"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("ANN Predicted RUL")
plt.title("ANN: Actual vs Predicted RUL")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "04_ann_actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# 6. ACTUAL VS PREDICTED - CNN
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    predictions["actual_RUL"],
    predictions["CNN_prediction"],
    alpha=0.35,
    s=15
)

min_value = min(
    predictions["actual_RUL"].min(),
    predictions["CNN_prediction"].min()
)

max_value = max(
    predictions["actual_RUL"].max(),
    predictions["CNN_prediction"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("CNN Predicted RUL")
plt.title("CNN: Actual vs Predicted RUL")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "05_cnn_actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# 7. ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    predictions["ANN_error"],
    bins=40,
    alpha=0.6,
    label="ANN"
)

plt.hist(
    predictions["CNN_error"],
    bins=40,
    alpha=0.6,
    label="CNN"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("ANN vs CNN Prediction Error Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "06_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 8. ABSOLUTE ERROR COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    predictions["ANN_abs_error"],
    bins=40,
    alpha=0.6,
    label="ANN"
)

plt.hist(
    predictions["CNN_abs_error"],
    bins=40,
    alpha=0.6,
    label="CNN"
)

plt.xlabel("Absolute Prediction Error (cycles)")
plt.ylabel("Frequency")
plt.title("ANN vs CNN Absolute Error Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "07_absolute_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 9. MODEL DISAGREEMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    predictions["ANN_CNN_disagreement"],
    bins=40
)

plt.xlabel("|ANN Prediction - CNN Prediction|")
plt.ylabel("Frequency")
plt.title("ANN-CNN Prediction Disagreement")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "08_ann_cnn_disagreement.png",
    dpi=200
)

plt.close()


# ============================================================
# 10. DISAGREEMENT VS ACTUAL ERROR
# ============================================================

predictions["mean_absolute_error"] = (
    predictions["ANN_abs_error"]
    + predictions["CNN_abs_error"]
) / 2


plt.figure(figsize=(8, 6))

plt.scatter(
    predictions["ANN_CNN_disagreement"],
    predictions["mean_absolute_error"],
    alpha=0.4,
    s=15
)

plt.xlabel("ANN-CNN Disagreement (cycles)")
plt.ylabel("Mean Absolute Error (cycles)")
plt.title("Model Disagreement vs Prediction Error")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "09_disagreement_vs_error.png",
    dpi=200
)

plt.close()


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

pearson = predictions[
    [
        "ANN_CNN_disagreement",
        "mean_absolute_error"
    ]
].corr(method="pearson").iloc[0, 1]

spearman = predictions[
    [
        "ANN_CNN_disagreement",
        "mean_absolute_error"
    ]
].corr(method="spearman").iloc[0, 1]


# ============================================================
# 12. HIGH-DISAGREEMENT OBSERVATIONS
# ============================================================

q75 = predictions[
    "ANN_CNN_disagreement"
].quantile(0.75)

high_disagreement = predictions[
    predictions["ANN_CNN_disagreement"] >= q75
].copy()

high_disagreement_mean_error = (
    high_disagreement["mean_absolute_error"].mean()
)

overall_mean_error = (
    predictions["mean_absolute_error"].mean()
)


# ============================================================
# 13. TRAINING CURVES - ANN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ann_history["loss"],
    label="Training Loss"
)

plt.plot(
    ann_history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "10_ann_training_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# 14. TRAINING CURVES - CNN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cnn_history["loss"],
    label="Training Loss"
)

plt.plot(
    cnn_history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "11_cnn_training_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# 15. CREATE FINAL SUMMARY REPORT
# ============================================================

summary_path = RESULTS_DIR / "student2_summary.txt"

with open(summary_path, "w") as f:

    f.write("AEROPULSE AI - STUDENT 2 SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("MODELS\n")
    f.write("-" * 60 + "\n")
    f.write("ANN: Fully connected neural network\n")
    f.write("CNN: 1D convolutional neural network\n\n")

    f.write("DATA\n")
    f.write("-" * 60 + "\n")
    f.write("Test observations: 2251\n")
    f.write("Input features: 17\n")
    f.write("Target: RUL (Remaining Useful Life)\n\n")

    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 60 + "\n")

    for _, row in comparison.iterrows():
        f.write(
            f"{row['Model']}:\n"
            f"  MAE  = {row['MAE']:.4f} cycles\n"
            f"  RMSE = {row['RMSE']:.4f} cycles\n"
            f"  R2   = {row['R2']:.4f}\n\n"
        )

    f.write("DISAGREEMENT ANALYSIS\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"Mean disagreement: "
        f"{predictions['ANN_CNN_disagreement'].mean():.4f} cycles\n"
    )

    f.write(
        f"Median disagreement: "
        f"{predictions['ANN_CNN_disagreement'].median():.4f} cycles\n"
    )

    f.write(
        f"Maximum disagreement: "
        f"{predictions['ANN_CNN_disagreement'].max():.4f} cycles\n"
    )

    f.write(
        f"75th percentile disagreement: "
        f"{q75:.4f} cycles\n\n"
    )

    f.write("ERROR / DISAGREEMENT RELATIONSHIP\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Overall mean absolute error across ANN/CNN: "
        f"{overall_mean_error:.4f} cycles\n"
    )

    f.write(
        f"Mean error for top 25% disagreement observations: "
        f"{high_disagreement_mean_error:.4f} cycles\n"
    )

    f.write(
        f"Pearson correlation: "
        f"{pearson:.4f}\n"
    )

    f.write(
        f"Spearman correlation: "
        f"{spearman:.4f}\n\n"
    )

    f.write("OUTPUT FILES\n")
    f.write("-" * 60 + "\n")
    f.write("student2_final_handoff.csv\n")
    f.write("student2_model_comparison_clean.csv\n")
    f.write("student2_summary.txt\n")
    f.write("plots/01_model_mae_comparison.png\n")
    f.write("plots/02_model_rmse_comparison.png\n")
    f.write("plots/03_model_r2_comparison.png\n")
    f.write("plots/04_ann_actual_vs_predicted.png\n")
    f.write("plots/05_cnn_actual_vs_predicted.png\n")
    f.write("plots/06_error_distribution.png\n")
    f.write("plots/07_absolute_error_distribution.png\n")
    f.write("plots/08_ann_cnn_disagreement.png\n")
    f.write("plots/09_disagreement_vs_error.png\n")
    f.write("plots/10_ann_training_curve.png\n")
    f.write("plots/11_cnn_training_curve.png\n")


# ============================================================
# 16. FINAL CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("AEROPULSE AI - STUDENT 2 COMPLETE")
print("=" * 60)

print("\nModels:")
print("  ANN + CNN")

print("\nTest observations:")
print(f"  {len(predictions)}")

print("\nANN:")
print(f"  MAE  = {comparison.loc[0, 'MAE']:.4f}")
print(f"  RMSE = {comparison.loc[0, 'RMSE']:.4f}")
print(f"  R2   = {comparison.loc[0, 'R2']:.4f}")

print("\nCNN:")
print(f"  MAE  = {comparison.loc[1, 'MAE']:.4f}")
print(f"  RMSE = {comparison.loc[1, 'RMSE']:.4f}")
print(f"  R2   = {comparison.loc[1, 'R2']:.4f}")

print("\nANN-CNN disagreement:")
print(
    f"  Mean   = "
    f"{predictions['ANN_CNN_disagreement'].mean():.4f}"
)

print(
    f"  Median = "
    f"{predictions['ANN_CNN_disagreement'].median():.4f}"
)

print(
    f"  Max    = "
    f"{predictions['ANN_CNN_disagreement'].max():.4f}"
)

print("\nCorrelation between disagreement and mean absolute error:")
print(f"  Pearson  = {pearson:.4f}")
print(f"  Spearman = {spearman:.4f}")

print("\nCreated:")
print(f"  {handoff_path}")
print(f"  {summary_path}")
print(f"  {PLOTS_DIR}/")

print("\n" + "=" * 60)
print("READY FOR STUDENT 4 HANDOFF")
print("=" * 60)
