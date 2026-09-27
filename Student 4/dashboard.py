"""
AeroPulse AI - Student 4 - Step 2: The Dashboard
==================================================

WHAT THIS FILE DOES (in plain English):

  This is the actual app people will click around in. It reads the
  file created by combine_predictions.py and shows it as an
  interactive webpage:

    - Pick an engine from a dropdown
    - See its predicted Remaining Useful Life (RUL) from all 4 models
    - See a big colored badge: Healthy / Warning / Critical
    - See the "disagreement score" (how much the 4 models argue)
    - See sensor readings over time for that engine
    - See which sensors matter most overall
    - See, across ALL engines, whether disagreement lines up with error
      (this is your project's main chart)

HOW TO RUN THIS:
  1. First run:  python combine_predictions.py
     (this creates the file this dashboard reads)
  2. Then run:   streamlit run dashboard.py
  3. It opens a browser tab automatically.

You do not need to understand Streamlit deeply. Every "st.something"
line below just draws one piece of the webpage - a title, a chart,
a dropdown, etc. Read the comments if you want to explain any of it
in your viva/demo.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================================
# PAGE SETUP
# =====================================================================

st.set_page_config(page_title="AeroPulse AI Dashboard", layout="wide")

st.title("AeroPulse AI")
st.caption(
    "Predictive maintenance dashboard combining ANN, CNN, LSTM and "
    "CNN-LSTM predictions of engine Remaining Useful Life (RUL)."
)


# =====================================================================
# LOAD DATA
# =====================================================================

@st.cache_data
def load_data():
    combined = pd.read_csv("combined_dashboard_data.csv")
    sensors = pd.read_csv("../Student 1/test_FD001.txt", sep=r"\s+", header=None)
    # Give the raw sensor file proper column names (same layout Student 1 used)
    columns = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    sensors.columns = columns
    importance = pd.read_csv("sensor_importance.csv")
    return combined, sensors, importance

try:
    combined, sensors, importance = load_data()
except FileNotFoundError as e:
    st.error(
        "Could not find the data files. Run 'python combine_predictions.py' "
        "first, in this same folder, then reload this page."
    )
    st.stop()


# =====================================================================
# SIDEBAR - pick an engine
# =====================================================================

st.sidebar.header("Select an engine")
engine_list = sorted(combined["unit"].unique())
selected_unit = st.sidebar.selectbox("Engine (unit) number", engine_list)

engine_data = combined[combined["unit"] == selected_unit].sort_values("cycle")
latest = engine_data.iloc[-1]  # most recent reading for this engine


# =====================================================================
# TOP ROW - headline numbers for the selected engine
# =====================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Predicted RUL (avg of 4 models)", f"{latest['ensemble_prediction']:.0f} cycles")

with col2:
    st.metric("Model disagreement", f"{latest['disagreement_score']:.1f} cycles")

with col3:
    st.metric("Actual RUL (ground truth)", f"{latest['actual_RUL']:.0f} cycles")

with col4:
    status = latest["health_status"]
    color = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}[status]
    st.metric("Health Status", f"{color} {status}")

if latest["health_status"] == "Critical":
    st.error(
        f"⚠️ Engine {selected_unit} is flagged CRITICAL. "
        "Recommend inspection / maintenance soon."
    )
elif latest["health_status"] == "Warning":
    st.warning(f"Engine {selected_unit} is flagged WARNING. Keep an eye on it.")
else:
    st.success(f"Engine {selected_unit} looks healthy.")


# =====================================================================
# ROW - the 4 individual model predictions, side by side
# =====================================================================

st.subheader("What each model predicted")
pred_table = engine_data[[
    "cycle", "ANN_prediction", "CNN_prediction", "LSTM_prediction",
    "CNNLSTM_prediction", "ensemble_prediction", "actual_RUL", "disagreement_score"
]].round(1)
st.dataframe(pred_table, use_container_width=True, hide_index=True)

fig_models = px.line(
    engine_data,
    x="cycle",
    y=["ANN_prediction", "CNN_prediction", "LSTM_prediction", "CNNLSTM_prediction", "actual_RUL"],
    labels={"value": "Predicted RUL (cycles)", "cycle": "Cycle", "variable": "Model"},
    title=f"Engine {selected_unit}: predictions over time vs actual RUL",
)
st.plotly_chart(fig_models, use_container_width=True)


# =====================================================================
# ROW - sensor trends for the selected engine
# =====================================================================

st.subheader("Sensor readings over time")
engine_sensors = sensors[sensors["unit"] == selected_unit].sort_values("cycle")
sensor_options = [c for c in engine_sensors.columns if c.startswith("s")]
chosen_sensors = st.multiselect(
    "Pick sensors to plot", sensor_options, default=sensor_options[:3]
)
if chosen_sensors:
    fig_sensors = px.line(
        engine_sensors, x="cycle", y=chosen_sensors,
        title=f"Engine {selected_unit}: raw sensor trends (degradation view)",
    )
    st.plotly_chart(fig_sensors, use_container_width=True)


# =====================================================================
# ROW - sensor importance (overall, across all engines)
# =====================================================================

st.subheader("Which sensors matter most (overall)")
fig_importance = px.bar(
    importance.head(10), x="importance_score", y="sensor", orientation="h",
    title="Top 10 sensors most correlated with Remaining Useful Life",
)
fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_importance, use_container_width=True)


# =====================================================================
# ROW - the project's main experiment, across ALL engines
# =====================================================================

st.subheader("Does model disagreement predict error? (main experiment)")
st.caption(
    "Each dot is one engine reading. If the dots trend upward left-to-right, "
    "it means: when the 4 models disagree more, they also tend to be more wrong. "
    "If the dots look like a random cloud, disagreement does NOT reliably predict error."
)
fig_scatter = px.scatter(
    combined, x="disagreement_score", y="ensemble_abs_error",
    color="health_status",
    color_discrete_map={"Healthy": "green", "Warning": "orange", "Critical": "red"},
    opacity=0.5,
    labels={
        "disagreement_score": "Model disagreement (cycles)",
        "ensemble_abs_error": "Actual error of average prediction (cycles)",
    },
)
st.plotly_chart(fig_scatter, use_container_width=True)

pearson = combined["disagreement_score"].corr(combined["ensemble_abs_error"])
st.write(f"**Correlation coefficient: {pearson:.3f}** "
         "(near 0 = weak relationship, near 1 = strong relationship)")
