# 🚀 AeroPulse AI

**A Group Project by:** Vedant, Pravesh, Sravan & Ishwari

## 📌 Overview
AeroPulse AI is a predictive maintenance system designed to analyze aircraft and jet engine sensor data. Using the NASA C-MAPSS dataset, it predicts the Remaining Useful Life (RUL) of an engine before failure occurs. 

## ⭐ Our Novel Approach: The Uncertainty Score
While predicting RUL using neural networks is common, our unique contribution focuses on **Model Disagreement**. We utilize four distinct models:
*   **ANN:** Learns basic relations between sensor values and engine life.
*   **CNN:** Identifies spatial patterns in sensor data.
*   **LSTM:** Understands time-based engine degradation.
*   **CNN-LSTM:** Captures both spatial patterns and time-based changes.

Instead of just picking the "best" model, we compare their outputs:
*   **High Agreement (e.g., 45, 44, 46, 45 cycles):** High confidence in the prediction.
*   **High Disagreement (e.g., 50, 31, 47, 29 cycles):** Flags an abnormal or uncertain engine condition.

We convert this disagreement into an **AI Agreement/Uncertainty Score** to act as an additional, vital health signal for maintenance crews.

## 🖥️ Dashboard Features
Built with **Streamlit**, our interactive dashboard displays:
*   Real-time Engine Health Status
*   Predicted Remaining Useful Life (RUL)
*   Individual Predictions from all 4 Models
*   **AI Agreement/Uncertainty Score**
*   Live Sensor Trends & Degradation Graphs
*   Automated Maintenance Warnings

## ⚙️ Tech Stack
*   **Dataset:** NASA C-MAPSS
*   **Machine Learning:** TensorFlow/Keras (ANN, CNN, LSTM, CNN-LSTM)
*   **Frontend/UI:** Streamlit
*   **Data Processing:** Pandas, NumPy, Scikit-learn
