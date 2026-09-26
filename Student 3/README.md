# AeroPulse AI — Student 3: LSTM & CNN-LSTM

## Overview

Student 3 implements time-series sequence modeling for jet engine RUL prediction using:
- **LSTM** (Long Short-Term Memory) 
- **CNN-LSTM** (Convolutional + LSTM Hybrid)

Both models use the exact same 17-feature dataset, 80/10/10 engine split, and StandardScaler normalization as Student 2's ANN and CNN — enabling fair comparison.

---

## Files

| File | Purpose |
|---|---|
| `student3_sequence_utils.py` | Reusable sliding-window sequence builder. Engine-boundary safe. |
| `student3_prepare_sequences.py` | Builds 3D sequence arrays from raw/split data. Run first. |
| `lstm_model.py` | LSTM model: build, train, evaluate, save. |
| `cnn_lstm_model.py` | CNN-LSTM model: build, train, evaluate, save. |
| `student3_evaluate.py` | Combined evaluation + 9 plots + Student 4 handoff CSV. |
| `student3_smoke_test.py` | Fast (~60s) pipeline validation. Run before full training. |

---

## How to Run (in order)

```bash
cd "Student 3"

# Step 0: Quick sanity check (optional but recommended)
python student3_smoke_test.py

# Step 1: Build sequence arrays
python student3_prepare_sequences.py

# Step 2: Train LSTM
python lstm_model.py

# Step 3: Train CNN-LSTM
python cnn_lstm_model.py

# Step 4: Evaluate + generate plots + create Student 4 handoff
python student3_evaluate.py
```

---

## Architecture

### LSTM

```
Input: (30 timesteps, 17 features)
       ↓
LSTM(64, return_sequences=True)
       ↓
Dropout(0.2)
       ↓
LSTM(32, return_sequences=False)
       ↓
Dropout(0.2)
       ↓
Dense(16, relu)
       ↓
Dense(1)  ← predicted RUL
```

### CNN-LSTM

```
Input: (30 timesteps, 17 features)
       ↓
Conv1D(64, kernel=3, padding=same, relu)   ← extract local patterns
       ↓
Conv1D(64, kernel=3, padding=same, relu)   ← deeper features
       ↓
MaxPooling1D(pool=2)                        ← downsample
       ↓
LSTM(64)                                    ← temporal memory
       ↓
Dropout(0.3)
       ↓
Dense(32, relu)
       ↓
Dense(1)  ← predicted RUL
```

---

## Data Details

| Property | Value |
|---|---|
| Dataset | C-MAPSS FD001 |
| Total engines | 100 |
| Train engines | 80 (IDs 1–80) |
| Val engines | 10 (IDs 81–90) |
| Test engines | 10 (IDs 91–100) |
| Window size | 30 cycles |
| Features | 17 (setting_1, setting_2, sensor_2/3/4/6/7/8/9/11/12/13/14/15/17/20/21) |
| Target | RUL = max_cycle − current_cycle |
| Scaling | StandardScaler (fit on train only) |
| Input shape | (30, 17) per sample |

---

## Training Setup

| Setting | Value |
|---|---|
| Loss function | MSE (same as ANN/CNN) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 128 |
| Max epochs | 100 |
| Early stopping patience | 10 epochs |
| LR reduction | ×0.5 when val_loss plateaus (patience=5) |
| Best model checkpoint | Saved automatically |

---

## Evaluation Metrics

Same metrics as Student 2: **MAE**, **RMSE**, **R²**

---

## Output Files

```
Student 3/
├── models/
│   ├── lstm_model.keras        ← trained LSTM
│   ├── lstm_best.keras         ← best checkpoint
│   ├── cnn_lstm_model.keras    ← trained CNN-LSTM
│   └── cnn_lstm_best.keras     ← best checkpoint
│
├── data/sequences/
│   ├── X_train_seq.npy    (N, 30, 17)
│   ├── y_train_seq.npy    (N,)
│   ├── X_val_seq.npy
│   ├── y_val_seq.npy
│   ├── X_test_seq.npy
│   ├── y_test_seq.npy
│   └── test_meta.csv      ← unit, cycle, actual_RUL per test sample
│
└── results/
    ├── lstm_predictions.csv          ← unit, cycle, actual_RUL, LSTM_prediction
    ├── cnn_lstm_predictions.csv      ← unit, cycle, actual_RUL, CNN_LSTM_prediction
    ├── lstm_metrics.csv
    ├── cnn_lstm_metrics.csv
    ├── lstm_training_history.csv
    ├── cnn_lstm_training_history.csv
    ├── student3_model_comparison.csv ← all 4 models: MAE, RMSE, R²
    ├── student3_final_handoff.csv    ← LSTM + CNN-LSTM predictions for Student 4
    └── plots/
        ├── 01_lstm_training_curve.png
        ├── 02_cnn_lstm_training_curve.png
        ├── 03_lstm_actual_vs_predicted.png
        ├── 04_cnn_lstm_actual_vs_predicted.png
        ├── 05_lstm_vs_cnn_lstm_error_dist.png
        ├── 06_all_models_mae.png
        ├── 07_all_models_rmse.png
        ├── 08_all_models_r2.png
        └── 09_example_engine_degradation.png
```

---

## For Student 4

### Files to load

```python
import pandas as pd

# Student 3's LSTM + CNN-LSTM predictions
s3_handoff = pd.read_csv("Student 3/results/student3_final_handoff.csv")
# Columns: unit, cycle, actual_RUL, LSTM_prediction, CNN_LSTM_prediction

# Student 2's ANN + CNN predictions  
s2_handoff = pd.read_csv("Student 2/student2_final_handoff.csv")
# Columns: unit, cycle, actual_RUL, ANN_prediction, CNN_prediction, ...

# Merge all 4 model predictions on (unit, cycle, actual_RUL)
all_preds = pd.merge(s2_handoff, s3_handoff, on=["unit", "cycle", "actual_RUL"], how="inner")
```

### Resulting DataFrame columns

```
unit | cycle | actual_RUL | ANN_prediction | CNN_prediction | LSTM_prediction | CNN_LSTM_prediction
```

Each row is traceable to a specific engine (`unit`) at a specific `cycle`, with the ground-truth `actual_RUL` and predictions from all 4 models.

> **Note on test-set alignment**: Student 3's test set uses 10 engines (IDs 91–100) with all valid 30-cycle windows. Student 2's ANN/CNN use the same 10 engines. The merge on `(unit, cycle, actual_RUL)` correctly aligns them since both use the same underlying split.

---

## Engine Boundary Guarantee

The `create_sequences()` function in `student3_sequence_utils.py` groups data by engine ID before creating windows. A window is **never** created that spans two different engines. This is verified in the smoke test (Step 5).

---

## Dependencies

- Python 3.8+
- TensorFlow / Keras ≥ 2.10
- NumPy, Pandas, scikit-learn, joblib, matplotlib
