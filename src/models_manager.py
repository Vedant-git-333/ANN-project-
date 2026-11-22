"""
AeroPulse AI - Model Manager, Cache, and Live Inference Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.constants import (
    FEATURES,
    SEQUENCE_LENGTH,
    ANN_MODEL_PATH,
    CNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    CNN_LSTM_MODEL_PATH,
    SCALER_PATH,
    DISAGREEMENT_LOW,
    DISAGREEMENT_HIGH,
    RUL_HEALTHY_THRESHOLD,
    RUL_CRITICAL_THRESHOLD,
)


@st.cache_resource(show_spinner="Loading AeroPulse AI Neural Network Suite...")
def load_models_and_scaler():
    """
    Loads and caches all four trained neural network models and the training StandardScaler.
    Cached across user interactions so models are not reloaded on slider adjustments.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Fitted training scaler not found at: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    models = {}

    # 1. Load ANN
    if os.path.exists(ANN_MODEL_PATH):
        models["ANN"] = tf.keras.models.load_model(ANN_MODEL_PATH)
    else:
        models["ANN"] = None

    # 2. Load CNN
    if os.path.exists(CNN_MODEL_PATH):
        models["CNN"] = tf.keras.models.load_model(CNN_MODEL_PATH)
    else:
        models["CNN"] = None

    # 3. Load LSTM
    if os.path.exists(LSTM_MODEL_PATH):
        models["LSTM"] = tf.keras.models.load_model(LSTM_MODEL_PATH)
    else:
        models["LSTM"] = None

    # 4. Load CNN-LSTM
    if os.path.exists(CNN_LSTM_MODEL_PATH):
        models["CNN_LSTM"] = tf.keras.models.load_model(CNN_LSTM_MODEL_PATH)
    else:
        models["CNN_LSTM"] = None

    return models, scaler


def predict_engine_dataframe(df: pd.DataFrame, models=None, scaler=None):
    """
    Executes end-to-end inference across all four models on any validated engine DataFrame:
    1. Preserves exact 17 feature order.
    2. Applies the pre-fitted training StandardScaler (no fitting on new data).
    3. Feeds (samples, 17) to ANN.
    4. Feeds (samples, 17, 1) to CNN.
    5. Builds (samples, 30, 17) sliding windows per engine (strictly respecting engine boundaries).
    6. Feeds 3D sequences to LSTM and CNN-LSTM.
    7. Aligns all predictions by unit and cycle.
    8. Computes ensemble average RUL, disagreement score, agreement category, and health status.
    """
    if models is None or scaler is None:
        models, scaler = load_models_and_scaler()

    results = []

    for unit_id, unit_df in df.groupby("unit"):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        raw_features = unit_df[FEATURES].values
        cycles = unit_df["cycle"].values
        n_obs = len(unit_df)

        # Scale features using training scaler
        scaled_features = scaler.transform(raw_features)

        # Point Model Predictions (ANN & CNN)
        # ANN expects (N, 17)
        if models["ANN"] is not None:
            ann_preds = models["ANN"].predict(scaled_features, verbose=0).flatten()
        else:
            ann_preds = np.full(n_obs, np.nan)

        # CNN expects (N, 17, 1)
        if models["CNN"] is not None:
            cnn_input = scaled_features.reshape(n_obs, 17, 1)
            cnn_preds = models["CNN"].predict(cnn_input, verbose=0).flatten()
        else:
            cnn_preds = np.full(n_obs, np.nan)

        # Sequence Model Predictions (LSTM & CNN-LSTM)
        # Require a 30-cycle temporal window
        lstm_preds = np.full(n_obs, np.nan)
        cnn_lstm_preds = np.full(n_obs, np.nan)

        if n_obs >= SEQUENCE_LENGTH:
            sequences = []
            valid_indices = []
            for i in range(n_obs - SEQUENCE_LENGTH + 1):
                window = scaled_features[i : i + SEQUENCE_LENGTH]
                sequences.append(window)
                # The prediction corresponds to the LAST cycle of the window
                valid_indices.append(i + SEQUENCE_LENGTH - 1)

            sequences_arr = np.array(sequences, dtype=np.float32)

            if models["LSTM"] is not None:
                p_lstm = models["LSTM"].predict(sequences_arr, verbose=0).flatten()
                for idx, p in zip(valid_indices, p_lstm):
                    lstm_preds[idx] = p

            if models["CNN_LSTM"] is not None:
                p_cnnlstm = models["CNN_LSTM"].predict(sequences_arr, verbose=0).flatten()
                for idx, p in zip(valid_indices, p_cnnlstm):
                    cnn_lstm_preds[idx] = p

        # Assemble per-cycle output for this engine
        for i in range(n_obs):
            row_dict = {
                "unit": unit_id,
                "cycle": cycles[i],
                "ANN_prediction": float(ann_preds[i]),
                "CNN_prediction": float(cnn_preds[i]),
                "LSTM_prediction": float(lstm_preds[i]) if not np.isnan(lstm_preds[i]) else np.nan,
                "CNNLSTM_prediction": float(cnn_lstm_preds[i]) if not np.isnan(cnn_lstm_preds[i]) else np.nan,
            }

            # If ground truth RUL is present in the source df
            if "actual_RUL" in unit_df.columns:
                row_dict["actual_RUL"] = float(unit_df.loc[i, "actual_RUL"])
            elif "RUL" in unit_df.columns:
                row_dict["actual_RUL"] = float(unit_df.loc[i, "RUL"])

            # Determine available model predictions
            active_preds = []
            for p_name in ["ANN_prediction", "CNN_prediction", "LSTM_prediction", "CNNLSTM_prediction"]:
                val = row_dict[p_name]
                if not np.isnan(val):
                    active_preds.append(val)

            # Compute ensemble metrics
            if len(active_preds) > 0:
                ens_pred = float(np.mean(active_preds))
                std_pred = float(np.std(active_preds, ddof=1)) if len(active_preds) > 1 else 0.0
                range_pred = float(np.max(active_preds) - np.min(active_preds))
            else:
                ens_pred = np.nan
                std_pred = np.nan
                range_pred = np.nan

            row_dict["ensemble_prediction"] = ens_pred
            row_dict["disagreement_score"] = std_pred
            row_dict["prediction_range"] = range_pred

            # Agreement tier
            if std_pred <= DISAGREEMENT_LOW:
                row_dict["agreement_tier"] = "HIGH AGREEMENT"
            elif std_pred <= DISAGREEMENT_HIGH:
                row_dict["agreement_tier"] = "MODERATE AGREEMENT"
            else:
                row_dict["agreement_tier"] = "HIGH DISAGREEMENT"

            # Health classification
            if not np.isnan(ens_pred):
                if ens_pred < RUL_CRITICAL_THRESHOLD:
                    row_dict["health_status"] = "Critical"
                elif ens_pred <= RUL_HEALTHY_THRESHOLD:
                    if std_pred > DISAGREEMENT_HIGH:
                        row_dict["health_status"] = "Critical"
                    else:
                        row_dict["health_status"] = "Warning"
                else:
                    if std_pred > DISAGREEMENT_HIGH:
                        row_dict["health_status"] = "Warning"
                    else:
                        row_dict["health_status"] = "Healthy"
            else:
                row_dict["health_status"] = "Unknown"

            # Retrospective error (if ground truth is present)
            if "actual_RUL" in row_dict and not np.isnan(ens_pred):
                row_dict["ensemble_abs_error"] = abs(ens_pred - row_dict["actual_RUL"])

            results.append(row_dict)

    return pd.DataFrame(results)
