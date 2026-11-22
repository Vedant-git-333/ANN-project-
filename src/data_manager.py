"""
AeroPulse AI - Data Management, Caching, and Validation Module
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    FEATURES,
    RAW_COLUMNS,
    RAW_TO_STANDARD_MAP,
    COMBINED_DATA_PATH,
    SENSOR_IMPORTANCE_PATH,
    MODEL_COMPARISON_PATH,
    RAW_TRAIN_PATH,
    RAW_TEST_PATH,
    SAMPLE_CSV_PATH,
    RUL_HEALTHY_THRESHOLD,
    RUL_CRITICAL_THRESHOLD,
    DISAGREEMENT_LOW,
    DISAGREEMENT_HIGH,
)


@st.cache_data
def load_combined_data():
    """Loads and caches the 1,961 aligned test evaluations from Student 4."""
    if not os.path.exists(COMBINED_DATA_PATH):
        raise FileNotFoundError(f"Combined data file not found at: {COMBINED_DATA_PATH}")
    df = pd.read_csv(COMBINED_DATA_PATH)
    return df


@st.cache_data
def load_sensor_importance():
    """Loads and caches precalculated sensor importance rankings."""
    if os.path.exists(SENSOR_IMPORTANCE_PATH):
        return pd.read_csv(SENSOR_IMPORTANCE_PATH)
    # Fallback default if file is missing
    return pd.DataFrame({
        "sensor": ["s11", "s4", "s12", "s7", "s15", "s21", "s20", "s17", "s2", "s3"],
        "importance_score": [0.775, 0.757, 0.749, 0.733, 0.721, 0.707, 0.705, 0.681, 0.678, 0.655]
    })


@st.cache_data
def load_model_metrics():
    """Loads and caches model evaluation benchmarks from Student 2 & 3."""
    if os.path.exists(MODEL_COMPARISON_PATH):
        return pd.read_csv(MODEL_COMPARISON_PATH)
    # Verified metrics recorded by Students 2 and 3
    return pd.DataFrame({
        "Model": ["ANN", "CNN", "LSTM", "CNN_LSTM"],
        "MAE": [37.6876, 39.5121, 31.1598, 31.1004],
        "RMSE": [53.2267, 56.1824, 49.3567, 47.5172],
        "R2": [0.5881, 0.5410, 0.5839, 0.6144]
    })


@st.cache_data
def load_c_mapss_raw():
    """Loads and caches the complete raw NASA C-MAPSS FD001 dataset with sensor values."""
    if not os.path.exists(RAW_TRAIN_PATH):
        raise FileNotFoundError(f"C-MAPSS raw training file not found at: {RAW_TRAIN_PATH}")
    
    df = pd.read_csv(RAW_TRAIN_PATH, sep=r"\s+", header=None, names=RAW_COLUMNS)
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["actual_RUL"] = max_cycles - df["cycle"]
    
    # Also standardize feature names for consistency
    for raw_c, std_c in RAW_TO_STANDARD_MAP.items():
        if raw_c in df.columns:
            df[std_c] = df[raw_c]
            
    return df


def get_sample_csv_bytes():
    """Reads or generates sample CSV data bytes for user download."""
    if os.path.exists(SAMPLE_CSV_PATH):
        with open(SAMPLE_CSV_PATH, "rb") as f:
            return f.read()
    
    # Generate on-the-fly from C-MAPSS
    df = load_c_mapss_raw()
    sample = df[df["unit"] == 95].sort_values("cycle").head(60)[["unit", "cycle"] + FEATURES].copy()
    return sample.to_csv(index=False).encode("utf-8")


def validate_uploaded_csv(df: pd.DataFrame):
    """
    Validates user-uploaded CSV file according to STEP 1 specification:
    - Check correct format and DataFrame instance
    - Check required columns (unit, cycle, 17 features, with automatic mapping from sX/opX if provided)
    - Check for missing values / NaNs
    - Check for non-numeric types
    - Check for duplicate engine-cycle records
    - Check sufficient observations
    
    Returns:
        (is_valid: bool, error_message: str, cleaned_df: pd.DataFrame or None)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "File could not be parsed as a valid tabular CSV.", None
    
    if df.empty:
        return False, "The uploaded CSV file is empty.", None
    
    # Check for unit / cycle identifiers (case-insensitive search)
    col_map = {c.lower().strip(): c for c in df.columns}
    
    unit_col = None
    for candidate in ["unit", "id", "engine", "engine_id", "unit_nr"]:
        if candidate in col_map:
            unit_col = col_map[candidate]
            break
            
    cycle_col = None
    for candidate in ["cycle", "cycles", "time", "time_in_cycles"]:
        if candidate in col_map:
            cycle_col = col_map[candidate]
            break
            
    if not unit_col or not cycle_col:
        return False, (
            "Missing required identifier columns: Your file must contain 'unit' (or 'id') "
            "and 'cycle' (or 'time_in_cycles') columns."
        ), None
        
    df_clean = df.copy()
    if unit_col != "unit":
        df_clean.rename(columns={unit_col: "unit"}, inplace=True)
    if cycle_col != "cycle":
        df_clean.rename(columns={cycle_col: "cycle"}, inplace=True)
        
    # Check if raw columns (s2, op1, etc.) were provided instead of setting_1, sensor_2
    for raw_name, std_name in RAW_TO_STANDARD_MAP.items():
        if std_name not in df_clean.columns and raw_name in df_clean.columns:
            df_clean[std_name] = df_clean[raw_name]
            
    # Verify all 17 features exist
    missing_features = [f for f in FEATURES if f not in df_clean.columns]
    if missing_features:
        return False, (
            f"Missing {len(missing_features)} required feature columns: {missing_features}. "
            "Please ensure your dataset contains all 17 operational settings and sensor measurements."
        ), None
        
    # Check for null / NaN values
    check_cols = ["unit", "cycle"] + FEATURES
    null_counts = df_clean[check_cols].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        return False, (
            f"Missing or NaN values detected in columns: {cols_with_nulls}. "
            "Neural network models require complete numeric observations."
        ), None
        
    # Check numeric types
    for c in check_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[c]):
            # Try to coerce to numeric
            try:
                df_clean[c] = pd.to_numeric(df_clean[c])
            except Exception:
                return False, f"Column '{c}' contains non-numeric values that cannot be processed.", None
                
    # Check for duplicate engine-cycle records
    duplicates = df_clean.duplicated(subset=["unit", "cycle"])
    if duplicates.any():
        num_dups = duplicates.sum()
        return False, (
            f"Detected {num_dups} duplicate (unit, cycle) records. "
            "Each engine observation must have a unique cycle timestamp."
        ), None
        
    # Sort chronologically
    df_clean = df_clean.sort_values(by=["unit", "cycle"]).reset_index(drop=True)
    
    # Check observation counts per engine
    obs_counts = df_clean.groupby("unit")["cycle"].count()
    if (obs_counts < 1).any():
        return False, "At least one engine has zero recorded observations.", None
        
    return True, "", df_clean


def compute_agreement(preds_series):
    """
    Computes agreement metrics from a 1D or row-wise array/Series of model predictions:
    - Mean (aggregate estimate)
    - Std (disagreement score)
    - Range (max - min)
    - Agreement Tier (HIGH, MODERATE, HIGH DISAGREEMENT)
    """
    arr = np.array(preds_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "category": "Unknown",
            "is_disagreement_alert": False
        }
        
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = max_val - min_val
    
    if std_val <= DISAGREEMENT_LOW:
        category = "HIGH AGREEMENT"
        is_alert = False
    elif std_val <= DISAGREEMENT_HIGH:
        category = "MODERATE AGREEMENT"
        is_alert = False
    else:
        category = "HIGH DISAGREEMENT"
        is_alert = True
        
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "category": category,
        "is_disagreement_alert": is_alert
    }


def classify_health_status(rul: float, disagreement: float = 0.0):
    """
    Multi-tier engine health classifier incorporating both aggregate RUL
    and model disagreement as an uncertainty penalty.
    """
    if pd.isna(rul):
        return "UNKNOWN", "Insufficient data to compute health status.", "gray"
        
    if rul < RUL_CRITICAL_THRESHOLD:
        status = "CRITICAL"
        rec = "Immediate maintenance and component boroscope inspection recommended."
        color = "red"
    elif rul <= RUL_HEALTHY_THRESHOLD:
        if disagreement > DISAGREEMENT_HIGH:
            status = "CRITICAL"
            rec = "Elevated model disagreement on degrading engine. Accelerated inspection schedule advised."
            color = "red"
        else:
            status = "DEGRADING"
            rec = "Standard degradation progression observed. Monitor sensor drift and plan overhaul."
            color = "orange"
    else:
        if disagreement > DISAGREEMENT_HIGH:
            status = "DEGRADING"
            rec = "High RUL estimate but significant model disparity detected. Cross-verify with vibration telemetry."
            color = "orange"
        else:
            status = "HEALTHY"
            rec = "All subsystem parameters nominal. Standard flight clearance approved."
            color = "green"
            
    return status, rec, color
