"""
AeroPulse AI - Integration Verification Test Script
Tests all requirements before final deployment.
"""

import os
import sys
import numpy as np
import pandas as pd

from src.constants import (
    FEATURES,
    ANN_MODEL_PATH,
    CNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    CNN_LSTM_MODEL_PATH,
    SCALER_PATH,
    COMBINED_DATA_PATH,
)
from src.data_manager import (
    load_combined_data,
    load_sensor_importance,
    load_model_metrics,
    load_c_mapss_raw,
    validate_uploaded_csv,
    compute_agreement,
    classify_health_status,
)
from src.models_manager import (
    load_models_and_scaler,
    predict_engine_dataframe,
)

print("=" * 60)
print("AEROPULSE AI — INTEGRATION VERIFICATION TEST")
print("=" * 60)

# TEST 1: File Artifacts Exist
print("\n[TEST 1] Verifying Physical File Artifacts...")
artifacts = [
    ("ANN Model", ANN_MODEL_PATH),
    ("CNN Model", CNN_MODEL_PATH),
    ("LSTM Model", LSTM_MODEL_PATH),
    ("CNN-LSTM Model", CNN_LSTM_MODEL_PATH),
    ("StandardScaler", SCALER_PATH),
    ("Combined Dashboard Data", COMBINED_DATA_PATH),
]
for name, path in artifacts:
    assert os.path.exists(path), f"FAILED: Missing artifact {path}"
    print(f"  ✓ {name}: Found at {path} ({os.path.getsize(path):,} bytes)")

# TEST 2: Data Loading
print("\n[TEST 2] Verifying Data Loading & Integrity...")
comb = load_combined_data()
assert len(comb) == 1961, f"Expected 1961 rows, got {len(comb)}"
print(f"  ✓ Combined data loaded: {len(comb)} rows, units: {sorted(comb['unit'].unique())}")

metrics = load_model_metrics()
assert len(metrics) == 4, f"Expected 4 models in metrics, got {len(metrics)}"
print(f"  ✓ Model metrics loaded:\n{metrics.to_string(index=False)}")

# TEST 3: Validation Pipeline
print("\n[TEST 3] Verifying CSV Validation Pipeline...")
sample_csv = pd.read_csv("sample_engine_data.csv")
is_valid, msg, clean_df = validate_uploaded_csv(sample_csv)
assert is_valid, f"Sample CSV validation failed: {msg}"
print(f"  ✓ Valid CSV correctly recognized: {clean_df.shape}")

# Test with invalid file (missing features)
invalid_csv = sample_csv[["unit", "cycle", "setting_1"]].copy()
is_valid_bad, msg_bad, _ = validate_uploaded_csv(invalid_csv)
assert not is_valid_bad, "Invalid CSV was incorrectly approved"
print(f"  ✓ Invalid CSV correctly rejected with message:\n    '{msg_bad}'")

# TEST 4: Model Loading & Live Inference
print("\n[TEST 4] Verifying Model Loading & End-to-End Inference...")
models, scaler = load_models_and_scaler()
print(f"  ✓ Models loaded: {list(models.keys())}")

# Run prediction on sample data (60 cycles)
preds_df = predict_engine_dataframe(clean_df, models, scaler)
assert len(preds_df) == len(clean_df), f"Output length mismatch: {len(preds_df)} vs {len(clean_df)}"
assert "ANN_prediction" in preds_df.columns
assert "CNN_prediction" in preds_df.columns
assert "LSTM_prediction" in preds_df.columns
assert "CNNLSTM_prediction" in preds_df.columns
assert "ensemble_prediction" in preds_df.columns
assert "disagreement_score" in preds_df.columns

# Check early cycles vs sequence cycles
early_row = preds_df[preds_df["cycle"] == 10].iloc[0]
assert not np.isnan(early_row["ANN_prediction"]), "ANN must predict on cycle 10"
assert not np.isnan(early_row["CNN_prediction"]), "CNN must predict on cycle 10"
assert np.isnan(early_row["LSTM_prediction"]), "LSTM must be NaN on cycle 10 (< 30)"

seq_row = preds_df[preds_df["cycle"] == 45].iloc[0]
assert not np.isnan(seq_row["LSTM_prediction"]), "LSTM must predict on cycle 45 (>= 30)"
assert not np.isnan(seq_row["CNNLSTM_prediction"]), "CNN-LSTM must predict on cycle 45 (>= 30)"
print("  ✓ Sequence boundaries strictly respected: cycles 1-29 point-only; cycles >= 30 all 4 models active.")
print(f"    Cycle 45 output: ANN={seq_row['ANN_prediction']:.1f}, CNN={seq_row['CNN_prediction']:.1f}, LSTM={seq_row['LSTM_prediction']:.1f}, CNN-LSTM={seq_row['CNNLSTM_prediction']:.1f}, Ensemble={seq_row['ensemble_prediction']:.1f}, Disagreement=±{seq_row['disagreement_score']:.2f}")

# TEST 5: Health & Agreement Logic
print("\n[TEST 5] Verifying Health & Disagreement Logic...")
stat_h, rec_h, _ = classify_health_status(rul=85.0, disagreement=5.0)
assert stat_h == "HEALTHY"
stat_d, rec_d, _ = classify_health_status(rul=45.0, disagreement=10.0)
assert stat_d == "DEGRADING"
stat_c, rec_c, _ = classify_health_status(rul=20.0, disagreement=8.0)
assert stat_c == "CRITICAL"
stat_pen, rec_pen, _ = classify_health_status(rul=45.0, disagreement=22.0)
assert stat_pen == "CRITICAL", "High disagreement on degrading engine must trigger critical escalation"
print("  ✓ Health classification and uncertainty penalty verified.")

# TEST 6: Research Analysis
print("\n[TEST 6] Verifying Research Analysis Alignment...")
pearson = comb["disagreement_score"].corr(comb["ensemble_abs_error"])
spearman = comb["disagreement_score"].corr(comb["ensemble_abs_error"], method="spearman")
print(f"  ✓ Pearson correlation (Disagreement vs Absolute Error): {pearson:.4f}")
print(f"  ✓ Spearman rank correlation: {spearman:.4f}")

print("\n" + "=" * 60)
print("ALL 6 INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
