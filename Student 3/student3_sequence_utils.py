"""
============================================================
AeroPulse AI - Student 3
FILE: student3_sequence_utils.py

PURPOSE:
    Reusable utilities for building sliding-window sequences
    from time-series engine data.

    These functions are used by both:
        - lstm_model.py
        - cnn_lstm_model.py

IMPORTANT:
    Engine boundaries are STRICTLY respected.
    A sequence is NEVER created that mixes rows from two
    different engines. This prevents data leakage and ensures
    the model truly learns temporal degradation patterns
    within a single engine.
============================================================
"""

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Window size: number of consecutive cycles per input sample.
# Student 1 already defined seq_length = 30 in Preprocess.py.
# We reuse the same value for a fair comparison.
WINDOW_SIZE = 30


# ============================================================
# CORE FUNCTION: create_sequences
# ============================================================

def create_sequences(df, window_size, feature_cols, rul_col="RUL"):
    """
    Build sliding-window sequences from a time-series DataFrame.

    Each sample consists of:
        - window_size consecutive rows of sensor/feature data
          from a SINGLE engine.
        - The RUL label at the LAST row of that window.

    This matches how Student 1 defined sequences in Preprocess.py.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'unit', 'cycle', all feature_cols,
        and rul_col.
        Rows must be sorted by unit then cycle (ascending).

    window_size : int
        Number of consecutive cycles to include in one sequence.
        Example: 30 -> input shape per sample = (30, num_features)

    feature_cols : list of str
        The sensor/feature column names to use as inputs.

    rul_col : str
        Name of the target column (default: "RUL").

    Returns
    -------
    X : np.ndarray, shape (n_samples, window_size, n_features)
        3D array of input sequences.

    y : np.ndarray, shape (n_samples,)
        1D array of RUL targets.

    meta : pd.DataFrame, shape (n_samples, 2)
        DataFrame with columns ['unit', 'cycle'] identifying
        which engine and which cycle each sample belongs to.
        The 'cycle' is the LAST cycle of the window (i.e., the
        current timestep for which RUL is being predicted).
        This is needed by Student 4 for traceability.
    """

    sequences = []   # will hold (window_size, n_features) arrays
    labels = []      # will hold scalar RUL values
    meta_rows = []   # will hold (unit, cycle) tuples

    # Process each engine independently to avoid cross-boundary windows
    for engine_id, engine_df in df.groupby("unit"):

        # Sort by cycle to guarantee chronological order
        engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

        feature_data = engine_df[feature_cols].values   # shape: (T, F)
        rul_data     = engine_df[rul_col].values         # shape: (T,)
        cycle_data   = engine_df["cycle"].values         # shape: (T,)

        n_cycles = len(feature_data)

        # Only create sequences if the engine has enough cycles
        if n_cycles < window_size:
            # Engine is shorter than window_size; skip it.
            # This is consistent with Student 1's generate_sequences().
            continue

        # Slide the window one step at a time
        for start in range(n_cycles - window_size + 1):
            end = start + window_size   # exclusive end index

            # X: window of sensor readings, shape (window_size, n_features)
            sequences.append(feature_data[start:end])

            # y: RUL at the LAST timestep of the window
            # (i.e., what is the remaining life at 'end - 1' cycle)
            labels.append(rul_data[end - 1])

            # meta: (engine_id, last_cycle_in_window) for traceability
            meta_rows.append((engine_id, cycle_data[end - 1]))

    # Convert to numpy arrays
    X = np.array(sequences, dtype=np.float32)   # (N, W, F)
    y = np.array(labels, dtype=np.float32)       # (N,)
    meta = pd.DataFrame(meta_rows, columns=["unit", "cycle"])

    return X, y, meta


# ============================================================
# HELPER: create_sequences_for_test
# ============================================================

def create_sequences_for_test(df, window_size, feature_cols, rul_col="RUL"):
    """
    Same as create_sequences() but explicitly named for the test split
    to make the calling code self-documenting.

    For test engines, we create sequences for every valid window.
    The aligned prediction output will give per-cycle RUL predictions
    for the test engines, which Student 4 can merge with ANN/CNN results.
    """
    return create_sequences(df, window_size, feature_cols, rul_col)


# ============================================================
# HELPER: verify_sequence_shapes
# ============================================================

def verify_sequence_shapes(X_train, y_train, X_val, y_val,
                           X_test, y_test, window_size, n_features):
    """
    Print a clear summary of all sequence array shapes.
    Confirms that:
      - X arrays are 3D: (samples, window_size, n_features)
      - y arrays are 1D: (samples,)
      - Dimensions match expectations

    Raises AssertionError if shapes are wrong.
    """
    print("\n========== SEQUENCE SHAPE VERIFICATION ==========")

    for name, X, y in [
        ("Train", X_train, y_train),
        ("Val  ", X_val,   y_val),
        ("Test ", X_test,  y_test),
    ]:
        assert X.ndim == 3, f"{name} X must be 3D, got {X.ndim}D"
        assert y.ndim == 1, f"{name} y must be 1D, got {y.ndim}D"
        assert X.shape[1] == window_size, (
            f"{name} X timesteps = {X.shape[1]}, expected {window_size}"
        )
        assert X.shape[2] == n_features, (
            f"{name} X features = {X.shape[2]}, expected {n_features}"
        )
        assert X.shape[0] == y.shape[0], (
            f"{name} sample count mismatch: X={X.shape[0]}, y={y.shape[0]}"
        )
        print(
            f"  {name}: X={X.shape}  y={y.shape}  OK"
        )

    print("=================================================\n")
