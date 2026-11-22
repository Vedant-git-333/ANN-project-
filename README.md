# 🚀 AeroPulse AI — Jet Engine Health Intelligence & RUL Prediction

**A Group Project by:** Vedant, Pravesh, Sravan & Ishwari  
**Final Year B.Tech AI / ANN Capstone Project**

---

## 📌 Executive Overview
**AeroPulse AI** is an aerospace-themed predictive maintenance intelligence system designed to analyze turbofan engine degradation telemetry using the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. 

The system unifies four distinct neural network architectures to estimate the **Remaining Useful Life (RUL)** of aircraft engines:
1. **ANN (Artificial Neural Network):** Instantaneous point regression across normalized sensor features.
2. **CNN (1D Convolutional Neural Network):** Channel-wise local feature extraction across sensor variables.
3. **LSTM (Long Short-Term Memory):** Recurrent temporal memory capturing historical degradation trajectories.
4. **CNN-LSTM (Hybrid Spatio-Temporal Network):** Hierarchical feature extraction combining convolutional filters with recurrent gating.

---

## ⭐ Core Research Contribution: Disagreement as an Uncertainty Signal
While standard deep learning projects merely report individual model metrics, our core research contribution investigates:
> *"Whether disagreement among multiple neural-network RUL predictions can provide an unsupervised, actionable signal associated with difficult or potentially uncertain engine-health predictions."*

### Empirical Findings (1,961 Held-Out Test Evaluations):
- **Pearson Linear Correlation:** $r = 0.2565$ ($p < 10^{-20}$)
- **Spearman Rank Correlation:** $\rho = 0.3631$ ($p < 10^{-20}$)
- **Low Disagreement ($\le 7.41$ cycles Std):** Mean prediction error = **18.10 cycles** (Median: 7.36 cycles)
- **High Disagreement ($> 15.04$ cycles Std):** Mean prediction error = **39.72 cycles** (Median: 30.89 cycles)
- **Key Takeaway:** High model divergence signals a **2.19× increase in actual prediction error**, proving that multi-architecture disagreement serves as an early triage indicator for aviation safety inspectors.

---

## 🖥️ Three Input Levels in the Dashboard

### Level 1 — Select an Engine
- Select any engine from the NASA C-MAPSS fleet.
- Dynamically derives actual available cycles, sensor history, and model predictions without hardcoding.

### Level 2 — Explore Engine Cycles
- Real-time interactive cycle slider: `[ min_cycle ─────●───── max_cycle ]`.
- Dynamically updates the current operating cycle, instantaneous vs. temporal models, RUL trajectory, health status, and sensor wear patterns.
- Clear distinction between **Retrospective Ground Truth RUL** and **Predicted RUL**.
- Early cycle protection: Cycles 1–29 display an informative notice that 30-cycle temporal sequence models require sufficient historical context.

### Level 3 — Upload Your Own CSV File ("Analyze Your Engine Data")
- Full drag-and-drop CSV uploader with one-click **Download Sample CSV** and **View Expected Structure**.
- **STEP 1 — File Validation:** Enforces correct format, required columns, missing value detection, non-numeric checks, and duplicate prevention.
- **STEP 2 — Preprocessing:** Applies the pre-fitted training `StandardScaler` (strict zero-leakage; no fitting on user data).
- **STEP 3 — Sequence Generation:** Assembles 30-cycle sliding windows respecting engine boundaries.
- **STEP 4 — Live Inference:** Executes all four trained models (ANN, CNN, LSTM, CNN-LSTM) live.
- **STEP 5 & 6 — Results & Triage:** Reports aggregate RUL, individual model estimates, model agreement badges, health status, and monitoring recommendations.

---

## 🚀 Quickstart & Launch Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser (default URL: `http://localhost:8501`).

### 3. Run Integration Verification Tests
```bash
python test_integration.py
```

---

## 📂 Repository Architecture
```text
├── app.py                            # Master Streamlit aerospace dashboard
├── requirements.txt                  # Python dependencies
├── sample_engine_data.csv            # Sample CSV for Level 3 upload testing
├── test_integration.py               # Comprehensive end-to-end verification script
│
├── src/                              # Modular system backend
│   ├── constants.py                  # Standard features, thresholds, and paths
│   ├── ui_theme.py                   # Aerospace dark mission-control styles & Plotly themes
│   ├── data_manager.py               # Dataset loading, caching, and CSV validation
│   └── models_manager.py             # Model caching, sequence builder, and live inference
│
├── Student 1/                        # Data preprocessing & initial feature selection
│   ├── train_FD001.txt               # Raw NASA C-MAPSS training runs (100 engines)
│   ├── test_FD001.txt                # Raw NASA C-MAPSS test telemetry
│   ├── RUL_FD001.txt                 # Ground truth final RUL labels
│   ├── train_clean_2D.csv            # Preprocessed 2D tabular dataset
│   └── Preprocess.py                 # 17-feature selection & normalization logic
│
├── Student 2/                        # Point models (ANN & CNN)
│   ├── models/                       # Trained ann_model.keras & cnn_model.keras
│   ├── ann_model.py                  # ANN architecture definition & training
│   ├── cnn_model.py                  # 1D CNN architecture definition & training
│   ├── student2_final_handoff.csv    # Student 2 predictions & agreement
│   └── features.txt                  # The 17 standardized feature names
│
├── Student 3/                        # Temporal sequence models (LSTM & CNN-LSTM)
│   ├── models/                       # Trained lstm_model.keras & cnn_lstm_model.keras
│   ├── models/student3_scaler.pkl    # Pre-fitted StandardScaler on training partition
│   ├── lstm_model.py                 # LSTM architecture definition & training
│   ├── cnn_lstm_model.py             # Hybrid CNN-LSTM definition & training
│   ├── student3_sequence_utils.py    # Boundary-safe sliding window generator (w=30)
│   └── results/student3_final_handoff.csv # Verified test predictions
│
└── Student 4/                        # Intelligence synthesis & initial prototype
    ├── combined_dashboard_data.csv   # 1,961 aligned test predictions (all 4 models)
    ├── sensor_importance.csv         # Feature correlation rankings
    └── combine_predictions.py        # Prediction merge & error alignment script
```
