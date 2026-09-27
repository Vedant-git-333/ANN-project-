"""
AeroPulse AI - Student 4 - Step 1: Combine everything
=======================================================

WHAT THIS FILE DOES (in plain English):

  1. Reads Student 1's raw test data (the sensor readings for each
     engine, at each point in time) so we have something to show
     on the dashboard graphs.
  2. Reads Student 2's predictions file (ANN and CNN guesses for how
     many cycles each engine has left).
  3. Reads (or, if not ready yet, FAKES) Student 3's predictions
     (LSTM and CNN-LSTM guesses).
  4. Puts all 4 models' guesses side by side in one table.
  5. For every row, works out:
       - the "average" guess of all 4 models
       - how much the 4 models DISAGREE with each other
         (this is the whole point of the project - if the models
         disagree a lot, that engine's prediction is less trustworthy)
       - a simple traffic-light HEALTH STATUS: Healthy / Warning / Critical
  6. Works out which sensors matter most for predicting RUL
     (just by checking which sensors move together with RUL the most -
     no fancy AI needed, just correlation, easy to explain in your report)
  7. Saves one clean CSV file that the Streamlit dashboard will read.

HOW TO USE THIS FILE:
  - Put this script inside your "Student 4" folder in the repo.
  - Make sure the paths below (in the CONFIG section) point to the
    real files from Student 1, 2, and 3.
  - Once Student 3 gives you their real prediction file, change
    USE_FAKE_STUDENT3_DATA to False and set LSTM_CNNLSTM_FILE to
    their file.
  - Run:  python combine_predictions.py
  - It will create "combined_dashboard_data.csv" - that is the file
    the dashboard reads.
"""

import pandas as pd
import numpy as np
import os

# =====================================================================
# CONFIG - change these paths if your folders are named differently
# =====================================================================

# Student 1's files
TEST_SENSOR_FILE = "../Student 1/test_FD001.txt"      # raw sensor readings
TRAIN_CLEAN_FILE = "../Student 1/train_clean_2D.csv"  # cleaned data, used for sensor importance

# Student 2's file (already has unit, cycle, actual_RUL, ANN_prediction, CNN_prediction)
ANN_CNN_FILE = "../Student 2/student2_final_handoff.csv"

# Student 3's real handoff file (unit, cycle, actual_RUL, LSTM_prediction, CNN_LSTM_prediction)
LSTM_CNNLSTM_FILE = "../Student 3/student3_final_handoff.csv"

# Student 3 has now uploaded real data, so we use it (no more fake numbers).
USE_FAKE_STUDENT3_DATA = False

# Where to save the final combined file
OUTPUT_FILE = "combined_dashboard_data.csv"
SENSOR_IMPORTANCE_FILE = "sensor_importance.csv"

# Health status thresholds - feel free to tune these numbers
RUL_HEALTHY_ABOVE = 60      # predicted RUL above this = Healthy
RUL_CRITICAL_BELOW = 30     # predicted RUL below this = Critical
DISAGREEMENT_HIGH = 15      # if models disagree by more than this many
                             # cycles, that pulls the status down a level


# =====================================================================
# STEP 1: Load Student 2's ANN + CNN predictions
# =====================================================================

print("Loading Student 2's ANN + CNN predictions...")
ann_cnn = pd.read_csv(ANN_CNN_FILE)
# Keep only the columns we actually need, so we don't get confused later
ann_cnn = ann_cnn[["unit", "cycle", "actual_RUL", "ANN_prediction", "CNN_prediction"]]
print(f"  -> {len(ann_cnn)} rows loaded")


# =====================================================================
# STEP 2: Load (or fake) Student 3's LSTM + CNN-LSTM predictions
# =====================================================================

if USE_FAKE_STUDENT3_DATA or not os.path.exists(LSTM_CNNLSTM_FILE):
    print("Student 3's file not found (or fake mode is on).")
    print("Creating PLACEHOLDER LSTM / CNN-LSTM predictions so the")
    print("pipeline runs end-to-end. Replace this with real data later!")

    np.random.seed(1)  # so the fake numbers are the same every time you run this
    lstm_cnnlstm = ann_cnn[["unit", "cycle"]].copy()

    # Fake predictions = ANN's prediction plus some random noise.
    # This is ONLY so you can test your code today. It is NOT real.
    lstm_cnnlstm["LSTM_prediction"] = (
        ann_cnn["ANN_prediction"] + np.random.normal(0, 8, len(ann_cnn))
    ).clip(lower=0)
    lstm_cnnlstm["CNNLSTM_prediction"] = (
        ann_cnn["CNN_prediction"] + np.random.normal(0, 8, len(ann_cnn))
    ).clip(lower=0)
else:
    print("Loading Student 3's real LSTM + CNN-LSTM predictions...")
    lstm_cnnlstm = pd.read_csv(LSTM_CNNLSTM_FILE)
    # Student 3's column is named "CNN_LSTM_prediction" (with an underscore) -
    # we rename it here so the rest of this script and the dashboard don't
    # need to care about that small naming difference.
    lstm_cnnlstm = lstm_cnnlstm.rename(columns={"CNN_LSTM_prediction": "CNNLSTM_prediction"})
    lstm_cnnlstm = lstm_cnnlstm[["unit", "cycle", "LSTM_prediction", "CNNLSTM_prediction"]]

print(f"  -> {len(lstm_cnnlstm)} rows ready")


# =====================================================================
# STEP 3: Merge everything into one table (match rows by engine + cycle)
# =====================================================================

print("Merging all 4 models' predictions together...")
combined = ann_cnn.merge(lstm_cnnlstm, on=["unit", "cycle"], how="inner")
print(f"  -> {len(combined)} rows after merging (rows that exist in all files)")

model_cols = ["ANN_prediction", "CNN_prediction", "LSTM_prediction", "CNNLSTM_prediction"]


# =====================================================================
# STEP 4: Work out the average prediction and the disagreement score
# =====================================================================

print("Calculating average prediction and model agreement/disagreement...")

# The "team's final guess" = simple average of the 4 models
combined["ensemble_prediction"] = combined[model_cols].mean(axis=1)

# Disagreement score = standard deviation across the 4 models.
# Small number = models mostly agree = more trustworthy.
# Big number = models are all over the place = less trustworthy.
combined["disagreement_score"] = combined[model_cols].std(axis=1)

# How far off was the average guess from the real answer? (for analysis)
combined["ensemble_abs_error"] = (combined["ensemble_prediction"] - combined["actual_RUL"]).abs()


# =====================================================================
# STEP 5: Turn all that into a simple traffic-light health status
# =====================================================================

def get_health_status(row):
    rul = row["ensemble_prediction"]
    disagreement = row["disagreement_score"]

    if rul < RUL_CRITICAL_BELOW:
        return "Critical"
    if rul < RUL_HEALTHY_ABOVE:
        # Borderline RUL - if the models also disagree a lot, be cautious
        if disagreement > DISAGREEMENT_HIGH:
            return "Critical"
        return "Warning"
    else:
        # RUL looks healthy, but flag it as Warning if models can't agree
        if disagreement > DISAGREEMENT_HIGH:
            return "Warning"
        return "Healthy"

combined["health_status"] = combined.apply(get_health_status, axis=1)

print("Health status counts:")
print(combined["health_status"].value_counts())


# =====================================================================
# STEP 6: Check your project's main question -
#         "Does disagreement between models predict error?"
# =====================================================================

pearson_corr = combined["disagreement_score"].corr(combined["ensemble_abs_error"], method="pearson")
spearman_corr = combined["disagreement_score"].corr(combined["ensemble_abs_error"], method="spearman")

print()
print("=" * 60)
print("MAIN EXPERIMENT RESULT")
print("=" * 60)
print(f"Pearson correlation (disagreement vs error):  {pearson_corr:.4f}")
print(f"Spearman correlation (disagreement vs error): {spearman_corr:.4f}")
print("(Close to 0 = weak relationship. Close to 1 = strong relationship.)")
print("This number is one of your key findings - write it in your report.")
print("=" * 60)


# =====================================================================
# STEP 7: Work out which sensors matter most (sensor importance)
# =====================================================================

print()
print("Calculating sensor importance...")
train_clean = pd.read_csv(TRAIN_CLEAN_FILE)

sensor_cols = [c for c in train_clean.columns if c.startswith("s") or c.startswith("op")]

# Simple, explainable method: correlation between each sensor and RUL.
# We use the absolute value because a strong NEGATIVE relationship
# (sensor goes up, RUL goes down) is just as important as a positive one.
importance = (
    train_clean[sensor_cols]
    .corrwith(train_clean["RUL"])
    .abs()
    .sort_values(ascending=False)
    .reset_index()
)
importance.columns = ["sensor", "importance_score"]
importance.to_csv(SENSOR_IMPORTANCE_FILE, index=False)

print("Top 5 most important sensors:")
print(importance.head(5).to_string(index=False))


# =====================================================================
# STEP 8: Save the final combined file for the dashboard
# =====================================================================

combined.to_csv(OUTPUT_FILE, index=False)
print()
print(f"DONE. Saved '{OUTPUT_FILE}' and '{SENSOR_IMPORTANCE_FILE}'.")


# =====================================================================
# STEP 9: Save some plots (PNG images) - matches the style Students 1-3
#         used, so your folder has visible results too, not just code.
# =====================================================================

import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

print("Saving plots to the 'plots' folder...")

# Plot 1: how many engines fall into each health status
plt.figure(figsize=(6, 4))
combined["health_status"].value_counts().reindex(["Healthy", "Warning", "Critical"]).plot(
    kind="bar", color=["green", "orange", "red"]
)
plt.title("Number of Readings by Health Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_health_status_counts.png", dpi=150)
plt.close()

# Plot 2: the project's main experiment - disagreement vs actual error
plt.figure(figsize=(6, 4))
plt.scatter(combined["disagreement_score"], combined["ensemble_abs_error"], alpha=0.3, s=10)
plt.xlabel("Model Disagreement (cycles)")
plt.ylabel("Actual Error of Ensemble Prediction (cycles)")
plt.title(f"Disagreement vs Error (Pearson r = {pearson_corr:.2f})")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_disagreement_vs_error.png", dpi=150)
plt.close()

# Plot 3: top 10 most important sensors
plt.figure(figsize=(6, 4))
top10 = importance.head(10).sort_values("importance_score")
plt.barh(top10["sensor"], top10["importance_score"], color="steelblue")
plt.xlabel("Correlation with RUL (absolute value)")
plt.title("Top 10 Most Important Sensors")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_sensor_importance.png", dpi=150)
plt.close()

# Plot 4: all 4 models' predictions vs actual RUL (all engines combined)
plt.figure(figsize=(6, 4))
plt.scatter(combined["actual_RUL"], combined["ensemble_prediction"], alpha=0.3, s=10)
max_val = max(combined["actual_RUL"].max(), combined["ensemble_prediction"].max())
plt.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
plt.xlabel("Actual RUL")
plt.ylabel("Ensemble (average) Predicted RUL")
plt.title("Ensemble Prediction vs Actual RUL")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_ensemble_vs_actual.png", dpi=150)
plt.close()

print("  -> saved 4 plots")


# =====================================================================
# STEP 10: Save a short plain-text summary (like Student 2's summary.txt)
# =====================================================================

with open("student4_summary.txt", "w") as f:
    f.write("STUDENT 4 - INTELLIGENCE + DASHBOARD - SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total readings combined (all 4 models present): {len(combined)}\n\n")
    f.write("Health status breakdown:\n")
    f.write(combined["health_status"].value_counts().to_string() + "\n\n")
    f.write("Main experiment - does disagreement predict error?\n")
    f.write(f"  Pearson correlation:  {pearson_corr:.4f}\n")
    f.write(f"  Spearman correlation: {spearman_corr:.4f}\n\n")
    f.write("Top 5 most important sensors:\n")
    f.write(importance.head(5).to_string(index=False) + "\n")

print("Saved 'student4_summary.txt'.")
print()
print("ALL DONE. Now run:  streamlit run dashboard.py")
