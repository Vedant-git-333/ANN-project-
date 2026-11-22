"""
AeroPulse AI — Jet Engine Health Intelligence & Remaining Useful Life (RUL) Prediction
Production Streamlit Application — Final Refined Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# System modules
from src.constants import (
    FEATURES,
    SEQUENCE_LENGTH,
    RUL_HEALTHY_THRESHOLD,
    RUL_CRITICAL_THRESHOLD,
    DISAGREEMENT_LOW,
    DISAGREEMENT_HIGH,
    DISCLAIMER_TEXT,
)
from src.ui_theme import (
    apply_custom_css,
    apply_aerospace_theme,
    render_kpi_card,
)
from src.data_manager import (
    load_combined_data,
    load_sensor_importance,
    load_model_metrics,
    load_c_mapss_raw,
    get_sample_csv_bytes,
    validate_uploaded_csv,
    classify_health_status,
)
from src.models_manager import (
    load_models_and_scaler,
    predict_engine_dataframe,
)

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & HIGH-CONTRAST CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroPulse AI — Jet Engine Health Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "uploaded_predictions_df" not in st.session_state:
    st.session_state.uploaded_predictions_df = None
if "uploaded_raw_df" not in st.session_state:
    st.session_state.uploaded_raw_df = None
if "use_uploaded_data" not in st.session_state:
    st.session_state.use_uploaded_data = False
if "selected_unit" not in st.session_state:
    st.session_state.selected_unit = 91
if "selected_cycle" not in st.session_state:
    st.session_state.selected_cycle = 80
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "📊 Overview"

# -----------------------------------------------------------------------------
# 3. DATA CACHING
# -----------------------------------------------------------------------------
try:
    combined_test_df = load_combined_data()
    sensor_importance_df = load_sensor_importance()
    model_metrics_df = load_model_metrics()
    c_mapss_raw_df = load_c_mapss_raw()
except Exception as e:
    st.error(f"Error loading system telemetry: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. ACTIVE DATASET RESOLUTION
# -----------------------------------------------------------------------------
if st.session_state.use_uploaded_data and st.session_state.uploaded_predictions_df is not None:
    active_df = st.session_state.uploaded_predictions_df
    is_custom_dataset = True
    dataset_name_display = "Custom Uploaded Telemetry"
else:
    active_df = combined_test_df
    is_custom_dataset = False
    dataset_name_display = "NASA C-MAPSS FD001 (Held-out Test Set)"

available_units = sorted(active_df["unit"].unique())

# Ensure session_state.selected_unit is valid in the active dataset
if st.session_state.selected_unit not in available_units:
    st.session_state.selected_unit = available_units[0]

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & UPLOAD CALLOUT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E314B; margin-bottom: 16px;">
            <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.85rem; font-weight: 700; color: #28C5F5; letter-spacing: 1.5px; line-height: 1.1;">
                ✈️ AEROPULSE AI
            </div>
            <div style="font-size: 0.82rem; color: #B7C5D8; margin-top: 4px; font-weight: 500;">
                Jet Engine Health & RUL Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Source Indicator Card
    if is_custom_dataset:
        st.markdown(
            """
            <div style="background: rgba(40, 197, 245, 0.12); border: 1px solid #28C5F5; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #28C5F5; font-weight: 700; letter-spacing: 0.8px;">Active Dataset</div>
                <div style="font-size: 0.92rem; color: #F4F7FC; font-weight: 600; margin-top: 2px;">📁 User-Uploaded Data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: #111F33; border: 1px solid #1E314B; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #B7C5D8; font-weight: 600; letter-spacing: 0.8px;">Active Dataset</div>
                <div style="font-size: 0.92rem; color: #24C78E; font-weight: 600; margin-top: 2px;">🛰️ NASA C-MAPSS FD001</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dedicated prominent action button to jump directly to upload
    if st.button("📤 Upload New Dataset (CSV)", use_container_width=True, type="secondary"):
        st.session_state.nav_page = "📤 Upload Dataset"
        st.rerun()

    pages = [
        "📊 Overview",
        "🔍 Engine Explorer",
        "📈 Cycle Analysis",
        "📤 Upload Dataset",
        "⚖️ Model Comparison",
        "📡 Sensor Analytics",
        "🔬 Research Contribution",
        "ℹ️ About Project",
    ]

    # Radio navigation
    nav_index = pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0
    selected_page = st.radio(
        "MISSION NAVIGATION",
        pages,
        index=nav_index,
        key="radio_nav",
    )
    if selected_page != st.session_state.nav_page:
        st.session_state.nav_page = selected_page
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #28C5F5; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">
            NEURAL ENSEMBLE STATUS
        </div>
        <div style="font-size: 0.82rem; color: #B7C5D8; line-height: 1.6;">
            • ANN (Point Dense): <b style="color:#24C78E;">ONLINE</b><br>
            • CNN (1D Conv): <b style="color:#24C78E;">ONLINE</b><br>
            • LSTM (Recurrent): <b style="color:#24C78E;">ONLINE</b><br>
            • CNN-LSTM (Hybrid): <b style="color:#24C78E;">ONLINE</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.72rem; color: #B7C5D8; line-height: 1.4;">
            {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 6. UNIFIED ENGINE & CYCLE SELECTION TOOLBAR
# -----------------------------------------------------------------------------
def render_engine_cycle_toolbar(show_dataset_toggle=True):
    """
    Renders synchronized engine dropdown and cycle slider.
    Updates st.session_state reliably without stale or out-of-bounds keys.
    """
    with st.container(border=True):
        col_unit, col_cycle = st.columns([1, 2])

        # Callback when engine dropdown changes
        def on_unit_change():
            new_unit = st.session_state["unit_select_box"]
            st.session_state.selected_unit = new_unit
            u_df = active_df[active_df["unit"] == new_unit]
            st.session_state.selected_cycle = int(u_df["cycle"].max())

        # Callback when cycle slider moves
        def on_cycle_change():
            st.session_state.selected_cycle = st.session_state["cycle_slider_widget"]

        with col_unit:
            curr_unit = st.session_state.selected_unit
            if curr_unit not in available_units:
                curr_unit = available_units[0]
                st.session_state.selected_unit = curr_unit

            st.selectbox(
                f"Selected Engine Unit ({dataset_name_display}):",
                available_units,
                index=available_units.index(curr_unit),
                key="unit_select_box",
                on_change=on_unit_change,
            )

        unit_trajectory = active_df[active_df["unit"] == st.session_state.selected_unit].sort_values("cycle")
        min_c = int(unit_trajectory["cycle"].min())
        max_c = int(unit_trajectory["cycle"].max())

        # Clamp cycle safely
        if st.session_state.selected_cycle < min_c:
            st.session_state.selected_cycle = min_c
        elif st.session_state.selected_cycle > max_c:
            st.session_state.selected_cycle = max_c

        with col_cycle:
            st.slider(
                f"Operating Cycle Position (Engine #{st.session_state.selected_unit}):",
                min_value=min_c,
                max_value=max_c,
                value=st.session_state.selected_cycle,
                step=1,
                key="cycle_slider_widget",
                on_change=on_cycle_change,
            )

        # Dataset toggle if user uploaded custom data
        if show_dataset_toggle and st.session_state.uploaded_predictions_df is not None:
            col_tog1, col_tog2 = st.columns([2, 1])
            with col_tog1:
                use_uploaded = st.checkbox(
                    "📁 View User-Uploaded Engine Data instead of C-MAPSS Benchmark",
                    value=st.session_state.use_uploaded_data,
                    key="dataset_source_toggle",
                )
                if use_uploaded != st.session_state.use_uploaded_data:
                    st.session_state.use_uploaded_data = use_uploaded
                    st.session_state.selected_unit = sorted(
                        (st.session_state.uploaded_predictions_df if use_uploaded else combined_test_df)["unit"].unique()
                    )[0]
                    st.rerun()

    return unit_trajectory, min_c, max_c


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if st.session_state.nav_page == "📊 Overview":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">AEROPULSE AI — <span>MISSION OVERVIEW</span></h1>
            <div class="aero-subtitle">Multi-Architecture Neural Network Health Monitoring & Remaining Useful Life (RUL) Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Interactive Selection Controls
    unit_trajectory, min_cycle, max_cycle = render_engine_cycle_toolbar()
    current_cycle = st.session_state.selected_cycle
    selected_unit = st.session_state.selected_unit

    # Retrieve current cycle snapshot
    cycle_match = unit_trajectory[unit_trajectory["cycle"] == current_cycle]
    if cycle_match.empty:
        cycle_row = unit_trajectory.iloc[-1]
        current_cycle = int(cycle_row["cycle"])
    else:
        cycle_row = cycle_match.iloc[0]

    p_ann = cycle_row["ANN_prediction"]
    p_cnn = cycle_row["CNN_prediction"]
    p_lstm = cycle_row["LSTM_prediction"]
    p_cnnlstm = cycle_row["CNNLSTM_prediction"]
    ensemble_rul = cycle_row["ensemble_prediction"]
    disagreement_score = cycle_row["disagreement_score"]
    health_status = cycle_row["health_status"]

    has_actual_rul = "actual_RUL" in cycle_row and not pd.isna(cycle_row["actual_RUL"])
    actual_rul = float(cycle_row["actual_RUL"]) if has_actual_rul else None

    # Calculate spread
    valid_p = [p for p in [p_ann, p_cnn, p_lstm, p_cnnlstm] if not pd.isna(p)]
    spread = (max(valid_p) - min(valid_p)) if len(valid_p) > 1 else 0.0

    # 2. Four Prominent Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.markdown(
            render_kpi_card(
                "ENGINE INFORMATION",
                f"UNIT #{selected_unit}",
                f"Cycle {current_cycle} of {max_cycle} | {dataset_name_display}",
            ),
            unsafe_allow_html=True,
        )

    with col_k2:
        gt_sub = f"Ground Truth: <b>{actual_rul:.0f}</b> cycles" if has_actual_rul else "Ground Truth: <i>Actual RUL Unavailable</i>"
        st.markdown(
            render_kpi_card(
                "PREDICTED RUL (ENSEMBLE)",
                f"{ensemble_rul:.1f} <span style='font-size:1.1rem; color:#B7C5D8;'>cycles</span>",
                gt_sub,
            ),
            unsafe_allow_html=True,
        )

    with col_k3:
        if disagreement_score <= DISAGREEMENT_LOW:
            d_badge = '<span class="badge-agreement-high">HIGH AGREEMENT</span>'
        elif disagreement_score <= DISAGREEMENT_HIGH:
            d_badge = '<span class="badge-agreement-mod">MODERATE AGREEMENT</span>'
        else:
            d_badge = '<span class="badge-agreement-disagree">HIGH DISAGREEMENT</span>'

        st.markdown(
            render_kpi_card(
                "MODEL DISAGREEMENT (STD)",
                f"±{disagreement_score:.2f} <span style='font-size:1.1rem; color:#B7C5D8;'>cycles</span>",
                f"Range: {spread:.1f} cycles | {d_badge}",
            ),
            unsafe_allow_html=True,
        )

    with col_k4:
        status_text, rec_text, _ = classify_health_status(ensemble_rul, disagreement_score)
        if status_text == "CRITICAL":
            h_badge = '<span class="badge-critical">CRITICAL DEGRADATION</span>'
        elif status_text == "DEGRADING" or status_text == "Warning":
            h_badge = '<span class="badge-warning">DEGRADING / WARNING</span>'
        else:
            h_badge = '<span class="badge-healthy">HEALTHY OPERATION</span>'

        st.markdown(
            render_kpi_card(
                "HEALTH DIAGNOSIS",
                status_text,
                h_badge,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Two-Column Core Analysis: Section B (RUL Summary) & Section C (Model Agreement)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("B. RUL Summary & Model Predictions")
            st.markdown(
                f"""
                <div style="font-size:0.92rem; color:#B7C5D8; margin-bottom: 12px;">
                    Predictions generated for <b>Engine #{selected_unit}</b> at <b>Cycle {current_cycle}</b> across all four neural architectures:
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Bar Chart comparing predictions
            chart_models = ["ANN", "CNN", "LSTM", "CNN-LSTM", "Ensemble"]
            chart_vals = [p_ann, p_cnn, p_lstm, p_cnnlstm, ensemble_rul]
            chart_colors = ["#28C5F5", "#8B5CF6", "#24C78E", "#EC4899", "#F5B942"]

            if has_actual_rul:
                chart_models.append("Ground Truth")
                chart_vals.append(actual_rul)
                chart_colors.append("#F06472")

            bar_df = pd.DataFrame({
                "Architecture": chart_models,
                "Predicted_RUL": chart_vals,
                "Color": chart_colors,
            })

            fig_bar = px.bar(
                bar_df,
                x="Architecture",
                y="Predicted_RUL",
                color="Architecture",
                color_discrete_map={m: c for m, c in zip(chart_models, chart_colors)},
                text=bar_df["Predicted_RUL"].apply(lambda v: f"{v:.1f}" if not pd.isna(v) else "N/A"),
                title=f"Cycle {current_cycle} Prediction Comparison (RUL in Cycles)",
            )
            fig_bar.update_traces(textposition="outside")
            apply_aerospace_theme(fig_bar, height=330)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Detailed table
            pred_summary_rows = [
                {"Architecture": "ANN (Point Dense)", "Input Shape": "(17,)", "Predicted RUL": f"{p_ann:.1f} cycles"},
                {"Architecture": "CNN (1D Conv)", "Input Shape": "(17, 1)", "Predicted RUL": f"{p_cnn:.1f} cycles"},
                {"Architecture": "LSTM (Recurrent)", "Input Shape": "(30, 17)", "Predicted RUL": f"{p_lstm:.1f} cycles" if not pd.isna(p_lstm) else "N/A (< 30 cycles)"},
                {"Architecture": "CNN-LSTM (Hybrid)", "Input Shape": "(30, 17)", "Predicted RUL": f"{p_cnnlstm:.1f} cycles" if not pd.isna(p_cnnlstm) else "N/A (< 30 cycles)"},
                {"Architecture": "Ensemble (Aggregate Mean)", "Input Shape": "Multi-Model", "Predicted RUL": f"{ensemble_rul:.1f} cycles"},
            ]
            if has_actual_rul:
                pred_summary_rows.append({
                    "Architecture": "Ground Truth (Actual)",
                    "Input Shape": "Run-to-Failure",
                    "Predicted RUL": f"{actual_rul:.0f} cycles (Diff: {abs(ensemble_rul - actual_rul):.1f} cycles)",
                })
            st.dataframe(pd.DataFrame(pred_summary_rows), use_container_width=True, hide_index=True)

    with col_right:
        with st.container(border=True):
            st.subheader("C. Model Agreement & Maintenance Triage")

            # 3-Tier Agreement Diagnosis
            if disagreement_score <= DISAGREEMENT_LOW:
                st.markdown(
                    f"""
                    <div class="banner-success">
                        <b>✅ HIGH MODEL CONSENSUS (Std: ±{disagreement_score:.2f} cycles, Spread: {spread:.1f} cycles)</b><br>
                        All four neural network models are in tight agreement. The aggregate RUL estimate has high empirical consensus.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif disagreement_score <= DISAGREEMENT_HIGH:
                st.markdown(
                    f"""
                    <div class="banner-warning">
                        <b>⚠️ MODERATE AGREEMENT (Std: ±{disagreement_score:.2f} cycles, Spread: {spread:.1f} cycles)</b><br>
                        Moderate variance observed between instantaneous point models and recurrent sequence models. Standard operational monitoring recommended.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="banner-danger">
                        <b>🚨 HIGH DISAGREEMENT DETECTED (Std: ±{disagreement_score:.2f} cycles, Spread: {spread:.1f} cycles)</b><br>
                        Significant divergence detected across model architectures. In our empirical validation, elevated disagreement indicates heightened prediction difficulty. Secondary sensor inspection recommended.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="background:#08111F; border:1px solid #1E314B; border-radius:6px; padding:14px; margin-top:14px;">
                    <div style="font-size:0.8rem; font-weight:700; color:#28C5F5; text-transform:uppercase; letter-spacing:0.8px;">Maintenance Recommendation:</div>
                    <div style="font-size:0.95rem; color:#F4F7FC; margin-top:4px; line-height:1.5;">{rec_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Mathematical Summary
            st.markdown("<br>", unsafe_allow_html=True)
            m_s1, m_s2 = st.columns(2)
            with m_s1:
                st.metric("Prediction Range", f"{spread:.1f} cycles", help="Max prediction minus Min prediction")
            with m_s2:
                st.metric("Disagreement Score", f"±{disagreement_score:.2f}", help="Sample standard deviation across the 4 predictions")

            st.caption(
                "Note: Model disagreement measures algorithmic variance between spatial and temporal feature extractors; it serves as an early uncertainty indicator, not guaranteed component failure."
            )

    # 4. Full-Width Visual Analytics: RUL Trajectory & Sensor Trends
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("D. Visual Analytics & Degradation Trends")

        tab_rul, tab_sensor = st.tabs(["📉 RUL Degradation Trajectory", "📡 Sensor Telemetry Dynamics"])

        with tab_rul:
            fig_rul = go.Figure()

            # Plot Actual RUL if available
            if has_actual_rul:
                fig_rul.add_trace(
                    go.Scatter(
                        x=unit_trajectory["cycle"],
                        y=unit_trajectory["actual_RUL"],
                        name="Actual RUL (Ground Truth)",
                        line=dict(color="#F06472", width=2.5, dash="dash"),
                    )
                )

            # Plot Ensemble Prediction
            fig_rul.add_trace(
                go.Scatter(
                    x=unit_trajectory["cycle"],
                    y=unit_trajectory["ensemble_prediction"],
                    name="Ensemble Predicted RUL",
                    line=dict(color="#28C5F5", width=2.5),
                )
            )

            # Marker for current cycle
            fig_rul.add_trace(
                go.Scatter(
                    x=[current_cycle],
                    y=[ensemble_rul],
                    mode="markers",
                    name=f"Current Cycle ({current_cycle})",
                    marker=dict(size=13, color="#F5B942", symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
                )
            )

            fig_rul.update_layout(
                title=f"Engine #{selected_unit} — RUL Trajectory across Operating Life",
                xaxis_title="Operating Cycle",
                yaxis_title="Remaining Useful Life (Cycles)",
            )
            apply_aerospace_theme(fig_rul, height=380)
            st.plotly_chart(fig_rul, use_container_width=True)

        with tab_sensor:
            col_s_opt1, col_s_opt2 = st.columns([1, 2])
            with col_s_opt1:
                if is_custom_dataset and st.session_state.uploaded_raw_df is not None:
                    sensor_source_df = st.session_state.uploaded_raw_df
                else:
                    sensor_source_df = c_mapss_raw_df

                unit_raw_sensors = sensor_source_df[sensor_source_df["unit"] == selected_unit].sort_values("cycle")

                available_sensor_cols = [c for c in FEATURES if c in unit_raw_sensors.columns]
                chosen_sensor = st.selectbox(
                    "Select Telemetry Channel to Plot:",
                    available_sensor_cols,
                    index=available_sensor_cols.index("sensor_11") if "sensor_11" in available_sensor_cols else 0,
                    key="overview_sensor_picker",
                )

            with col_s_opt2:
                fig_s_trend = px.line(
                    unit_raw_sensors,
                    x="cycle",
                    y=chosen_sensor,
                    title=f"Engine #{selected_unit} — {chosen_sensor} Telemetry Progression",
                    color_discrete_sequence=["#28C5F5"],
                )
                if current_cycle in unit_raw_sensors["cycle"].values:
                    c_val = unit_raw_sensors[unit_raw_sensors["cycle"] == current_cycle][chosen_sensor].iloc[0]
                    fig_s_trend.add_trace(
                        go.Scatter(
                            x=[current_cycle],
                            y=[c_val],
                            mode="markers",
                            name=f"Cycle {current_cycle}",
                            marker=dict(size=12, color="#F06472", symbol="diamond"),
                        )
                    )
                apply_aerospace_theme(fig_s_trend, height=340)
                st.plotly_chart(fig_s_trend, use_container_width=True)


# =============================================================================
# PAGE 2: ENGINE EXPLORER (LEVEL 1 & LEVEL 2)
# =============================================================================
elif st.session_state.nav_page == "🔍 Engine Explorer":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">ENGINE EXPLORER — <span>FLEET TELEMETRY</span></h1>
            <div class="aero-subtitle">Level 1 Engine Selection & Level 2 Cycle Slider for Real-Time Degradation Inspection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    unit_trajectory, min_c, max_c = render_engine_cycle_toolbar()
    explorer_cycle = st.session_state.selected_cycle
    explorer_unit = st.session_state.selected_unit

    curr_rec = unit_trajectory[unit_trajectory["cycle"] == explorer_cycle].iloc[0]

    # Metrics Summary
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Selected Engine", f"Unit #{explorer_unit}")
    with m2:
        st.metric("Inspected Cycle", f"Cycle {explorer_cycle} / {max_c}")
    with m3:
        st.metric("Ensemble Predicted RUL", f"{curr_rec['ensemble_prediction']:.1f} cycles")
    with m4:
        st.metric("Disagreement (Std Dev)", f"±{curr_rec['disagreement_score']:.2f} cycles")

    # Four Model Output Breakdown Cards
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader(f"Cycle {explorer_cycle} Multi-Model Output Breakdown")
        cp1, cp2, cp3, cp4 = st.columns(4)

        has_gt = "actual_RUL" in curr_rec and not pd.isna(curr_rec["actual_RUL"])
        gt_val = float(curr_rec["actual_RUL"]) if has_gt else None

        with cp1:
            err_txt = f"Error: {abs(curr_rec['ANN_prediction'] - gt_val):.1f} cycles" if has_gt else "Point Model (Dense)"
            st.markdown(render_kpi_card("ANN PREDICTION", f"{curr_rec['ANN_prediction']:.1f}", err_txt), unsafe_allow_html=True)
        with cp2:
            err_txt = f"Error: {abs(curr_rec['CNN_prediction'] - gt_val):.1f} cycles" if has_gt else "Point Model (1D Conv)"
            st.markdown(render_kpi_card("CNN PREDICTION", f"{curr_rec['CNN_prediction']:.1f}", err_txt), unsafe_allow_html=True)
        with cp3:
            if pd.isna(curr_rec["LSTM_prediction"]):
                lstm_val_str = "N/A"
                lstm_sub = "Requires 30-cycle window"
            else:
                lstm_val_str = f"{curr_rec['LSTM_prediction']:.1f}"
                lstm_sub = f"Error: {abs(curr_rec['LSTM_prediction'] - gt_val):.1f} cycles" if has_gt else "Temporal Recurrent"
            st.markdown(render_kpi_card("LSTM PREDICTION", lstm_val_str, lstm_sub), unsafe_allow_html=True)
        with cp4:
            if pd.isna(curr_rec["CNNLSTM_prediction"]):
                cnnlstm_val_str = "N/A"
                cnnlstm_sub = "Requires 30-cycle window"
            else:
                cnnlstm_val_str = f"{curr_rec['CNNLSTM_prediction']:.1f}"
                cnnlstm_sub = f"Error: {abs(curr_rec['CNNLSTM_prediction'] - gt_val):.1f} cycles" if has_gt else "Hybrid Spatio-Temporal"
            st.markdown(render_kpi_card("CNN-LSTM PREDICTION", cnnlstm_val_str, cnnlstm_sub), unsafe_allow_html=True)

    # Full Degradation Trajectory Plot
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader(f"Engine #{explorer_unit} Trajectory: All Models Across Operating Life")
        fig_multi = go.Figure()

        if has_gt:
            fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["actual_RUL"], name="Actual RUL (Ground Truth)", line=dict(color="#F06472", width=2.5, dash="dash")))

        fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["ANN_prediction"], name="ANN Prediction", line=dict(color="#28C5F5", width=1.5)))
        fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["CNN_prediction"], name="CNN Prediction", line=dict(color="#8B5CF6", width=1.5)))
        fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["LSTM_prediction"], name="LSTM Prediction", line=dict(color="#24C78E", width=1.5)))
        fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["CNNLSTM_prediction"], name="CNN-LSTM Prediction", line=dict(color="#EC4899", width=2)))
        fig_multi.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["ensemble_prediction"], name="Ensemble Average", line=dict(color="#F5B942", width=3)))

        # Highlight current cycle with vertical dashed line
        fig_multi.add_vline(x=explorer_cycle, line_width=1.5, line_dash="dash", line_color="#FFFFFF")

        fig_multi.update_layout(
            title=f"Engine #{explorer_unit} Lifetime Projections (Marker at Cycle {explorer_cycle})",
            xaxis_title="Operating Cycles",
            yaxis_title="Remaining Useful Life (Cycles)",
        )
        apply_aerospace_theme(fig_multi, height=430)
        st.plotly_chart(fig_multi, use_container_width=True)


# =============================================================================
# PAGE 3: CYCLE ANALYSIS
# =============================================================================
elif st.session_state.nav_page == "📈 Cycle Analysis":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">CYCLE-WISE ANALYSIS & <span>ERROR DYNAMICS</span></h1>
            <div class="aero-subtitle">Cycle-by-Cycle Model Projections, Disagreement Evolution, and Error Tracking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    unit_trajectory, min_c, max_c = render_engine_cycle_toolbar()
    sel_u = st.session_state.selected_unit

    has_ground_truth = "actual_RUL" in unit_trajectory.columns and not unit_trajectory["actual_RUL"].isnull().all()

    tab_t, tab_e, tab_d = st.tabs(["📊 Multi-Model Trajectory", "📉 Error Dynamics Over Time", "⚡ Disagreement Score Evolution"])

    with tab_t:
        fig_t = go.Figure()
        if has_ground_truth:
            fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["actual_RUL"], name="Actual RUL", line=dict(color="#F06472", width=3, dash="dot")))
        fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["ANN_prediction"], name="ANN", line=dict(color="#28C5F5", width=1.5)))
        fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["CNN_prediction"], name="CNN", line=dict(color="#8B5CF6", width=1.5)))
        fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["LSTM_prediction"], name="LSTM", line=dict(color="#24C78E", width=1.5)))
        fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["CNNLSTM_prediction"], name="CNN-LSTM", line=dict(color="#EC4899", width=2)))
        fig_t.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["ensemble_prediction"], name="Ensemble", line=dict(color="#F5B942", width=2.5)))
        apply_aerospace_theme(fig_t, height=440)
        st.plotly_chart(fig_t, use_container_width=True)

    with tab_e:
        if has_ground_truth:
            err_df = unit_trajectory.copy()
            err_df["ann_e"] = (err_df["ANN_prediction"] - err_df["actual_RUL"]).abs()
            err_df["cnn_e"] = (err_df["CNN_prediction"] - err_df["actual_RUL"]).abs()
            err_df["lstm_e"] = (err_df["LSTM_prediction"] - err_df["actual_RUL"]).abs()
            err_df["cnnlstm_e"] = (err_df["CNNLSTM_prediction"] - err_df["actual_RUL"]).abs()
            err_df["ens_e"] = (err_df["ensemble_prediction"] - err_df["actual_RUL"]).abs()

            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=err_df["cycle"], y=err_df["ann_e"], name="ANN Absolute Error", line=dict(color="#28C5F5", width=1.2)))
            fig_e.add_trace(go.Scatter(x=err_df["cycle"], y=err_df["cnn_e"], name="CNN Absolute Error", line=dict(color="#8B5CF6", width=1.2)))
            fig_e.add_trace(go.Scatter(x=err_df["cycle"], y=err_df["lstm_e"], name="LSTM Absolute Error", line=dict(color="#24C78E", width=1.2)))
            fig_e.add_trace(go.Scatter(x=err_df["cycle"], y=err_df["cnnlstm_e"], name="CNN-LSTM Absolute Error", line=dict(color="#EC4899", width=1.5)))
            fig_e.add_trace(go.Scatter(x=err_df["cycle"], y=err_df["ens_e"], name="Ensemble Absolute Error", line=dict(color="#F5B942", width=2.5)))
            fig_e.update_layout(title=f"Engine #{sel_u} — Absolute Prediction Error (|Predicted - Actual|)", xaxis_title="Operating Cycle", yaxis_title="Error (Cycles)")
            apply_aerospace_theme(fig_e, height=440)
            st.plotly_chart(fig_e, use_container_width=True)
        else:
            st.info("Actual Ground Truth RUL is unavailable for custom user-uploaded datasets. Retrospective error dynamics require run-to-failure labels.")

    with tab_d:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=unit_trajectory["cycle"], y=unit_trajectory["disagreement_score"], name="Disagreement (Std Dev)", line=dict(color="#F97316", width=2.5)))
        fig_d.add_hline(y=DISAGREEMENT_LOW, line_dash="dash", line_color="#24C78E", annotation_text="Tercile 1: Low Disagreement (<=7.41)")
        fig_d.add_hline(y=DISAGREEMENT_HIGH, line_dash="dash", line_color="#F06472", annotation_text="Tercile 3: High Disagreement (>15.04)")
        fig_d.update_layout(title=f"Engine #{sel_u} — Model Disagreement Progression Over Cycles", xaxis_title="Operating Cycle", yaxis_title="Standard Deviation Across Models")
        apply_aerospace_theme(fig_d, height=440)
        st.plotly_chart(fig_d, use_container_width=True)

    # Tabular Display
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader(f"Engine #{sel_u} Cycle Telemetry Table")
        display_cols = [c for c in ["cycle", "actual_RUL", "ANN_prediction", "CNN_prediction", "LSTM_prediction", "CNNLSTM_prediction", "ensemble_prediction", "disagreement_score", "health_status"] if c in unit_trajectory.columns]
        st.dataframe(unit_trajectory[display_cols].round(2), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 4: UPLOAD DATASET (LEVEL 3)
# =============================================================================
elif st.session_state.nav_page == "📤 Upload Dataset":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">ANALYZE YOUR ENGINE DATA — <span>CSV PREDICTION PIPELINE</span></h1>
            <div class="aero-subtitle">Upload your engine sensor history to generate AI-powered RUL predictions across all 4 trained neural models.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("1. Download Sample Template & Schema Specifications")
        st.markdown(
            """
            Upload your continuous aircraft engine sensor measurements in CSV format.
            The pipeline will validate columns, normalize features using the pre-fitted C-MAPSS training scaler,
            construct 30-cycle sliding sequences (without crossing engine boundaries), and execute live predictions.
            """
        )

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.download_button(
                label="📥 Download Compatible Sample CSV Template",
                data=get_sample_csv_bytes(),
                file_name="sample_engine_sensors.csv",
                mime="text/csv",
                help="Download a pre-formatted 60-cycle engine sensor file ready for instant testing.",
                use_container_width=True,
            )

        with col_b2:
            with st.expander("🔍 View Expected Column Schema & Requirements"):
                st.markdown(
                    f"""
                    **Required Identifiers:**
                    - `unit`: Integer engine identifier (e.g. 1, 2, 95)
                    - `cycle`: Positive integer time counter (1, 2, 3, ...)

                    **Required 17 Selected Features:**
                    `{', '.join(FEATURES)}`

                    *Note:* Raw NASA C-MAPSS column names (`op1`, `op2`, `s2`..`s21`) are also automatically mapped.
                    *Minimum Cycles:* Point models (ANN, CNN) predict from Cycle 1. Sequence models (LSTM, CNN-LSTM) require at least 30 consecutive cycles per engine.
                    """
                )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("2. Upload Your CSV File")

        uploaded_file = st.file_uploader(
            "Select or drop your sensor telemetry CSV file (.csv format supported):",
            type=["csv"],
            key="csv_file_uploader",
            help="Drag-and-drop or click Browse to upload your sensor history file.",
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Uploaded File Preview:** `{uploaded_file.name}` ({len(raw_df):,} rows, {len(raw_df.columns)} columns)")
                st.dataframe(raw_df.head(5), use_container_width=True)

                # Validation step
                st.markdown("### Data Validation Results")
                is_valid, err_msg, cleaned_upload = validate_uploaded_csv(raw_df)

                if not is_valid:
                    st.markdown(
                        f"""
                        <div class="banner-danger">
                            <b>❌ Validation Failed:</b> {err_msg}<br>
                            Please correct the dataset and re-upload. You can download the sample template above for reference.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    num_engines = cleaned_upload["unit"].nunique()
                    min_obs_per_engine = cleaned_upload.groupby("unit")["cycle"].count().min()
                    max_obs_per_engine = cleaned_upload.groupby("unit")["cycle"].count().max()

                    st.markdown(
                        f"""
                        <div class="banner-success">
                            <b>✅ Validation Passed:</b> Found <b>{num_engines}</b> engine(s) with {len(cleaned_upload):,} total observations.<br>
                            All 17 sensor features verified. Observation counts per engine range from <b>{min_obs_per_engine}</b> to <b>{max_obs_per_engine}</b> cycles.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if min_obs_per_engine < SEQUENCE_LENGTH:
                        st.markdown(
                            f"""
                            <div class="banner-warning">
                                <b>ℹ️ Sequence Length Notice:</b> Some engines have fewer than 30 cycles. Point models (ANN and CNN) will generate predictions from Cycle 1, while sequence models (LSTM and CNN-LSTM) will activate for cycles where $\\ge 30$ historical observations exist.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Prominent Predict RUL Button
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🚀 Predict Remaining Useful Life (RUL)", type="primary", use_container_width=True):
                        with st.spinner("Executing neural inference across ANN, CNN, LSTM, and CNN-LSTM..."):
                            models_suite, scaler_obj = load_models_and_scaler()
                            preds_df = predict_engine_dataframe(cleaned_upload, models_suite, scaler_obj)
                            st.session_state.uploaded_predictions_df = preds_df
                            st.session_state.uploaded_raw_df = cleaned_upload
                            st.session_state.use_uploaded_data = True
                            st.session_state.selected_unit = sorted(preds_df["unit"].unique())[0]
                            st.session_state.selected_cycle = int(preds_df[preds_df["unit"] == st.session_state.selected_unit]["cycle"].max())
                            st.success("✅ Prediction Complete! Results are ready below and active across all dashboard pages.")
                            st.rerun()

            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    # Display results if uploaded predictions exist
    if st.session_state.uploaded_predictions_df is not None:
        up_df = st.session_state.uploaded_predictions_df
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("3. Uploaded Telemetry — Prediction & Disagreement Results")

            up_units = sorted(up_df["unit"].unique())
            col_sel1, col_sel2 = st.columns([1, 2])

            with col_sel1:
                cur_up_unit = st.selectbox("Inspect Uploaded Engine:", up_units, key="upload_unit_selector")

            unit_up_preds = up_df[up_df["unit"] == cur_up_unit].sort_values("cycle")
            min_u_c = int(unit_up_preds["cycle"].min())
            max_u_c = int(unit_up_preds["cycle"].max())

            with col_sel2:
                cur_up_c = st.slider(f"Engine #{cur_up_unit} Cycle Slider:", min_value=min_u_c, max_value=max_u_c, value=max_u_c, step=1, key="upload_cycle_slider")

            row_u = unit_up_preds[unit_up_preds["cycle"] == cur_up_c].iloc[0]

            # Results KPIs
            uk1, uk2, uk3, uk4 = st.columns(4)
            with uk1:
                st.metric("Engine ID", f"Unit #{cur_up_unit}")
            with uk2:
                st.metric("Cycle Position", f"Cycle {cur_up_c} / {max_u_c}")
            with uk3:
                st.metric("Ensemble Predicted RUL", f"{row_u['ensemble_prediction']:.1f} cycles")
            with uk4:
                st.metric("Model Disagreement", f"±{row_u['disagreement_score']:.2f} cycles")

            # Model breakdown cards
            st.markdown("<br>", unsafe_allow_html=True)
            uc1, uc2, uc3, uc4 = st.columns(4)
            with uc1:
                v_ann = f"{row_u['ANN_prediction']:.1f} cycles" if not pd.isna(row_u['ANN_prediction']) else "N/A"
                st.markdown(render_kpi_card("ANN PREDICTION", v_ann, "Point Model (Dense)"), unsafe_allow_html=True)
            with uc2:
                v_cnn = f"{row_u['CNN_prediction']:.1f} cycles" if not pd.isna(row_u['CNN_prediction']) else "N/A"
                st.markdown(render_kpi_card("CNN PREDICTION", v_cnn, "Point Model (1D Conv)"), unsafe_allow_html=True)
            with uc3:
                v_lstm = f"{row_u['LSTM_prediction']:.1f} cycles" if not pd.isna(row_u['LSTM_prediction']) else "N/A (< 30 cycles)"
                st.markdown(render_kpi_card("LSTM PREDICTION", v_lstm, "Temporal Recurrent"), unsafe_allow_html=True)
            with uc4:
                v_cnnlstm = f"{row_u['CNNLSTM_prediction']:.1f} cycles" if not pd.isna(row_u['CNNLSTM_prediction']) else "N/A (< 30 cycles)"
                st.markdown(render_kpi_card("CNN-LSTM PREDICTION", v_cnnlstm, "Hybrid Spatio-Temporal"), unsafe_allow_html=True)

            # RUL Trajectory Plot for uploaded engine
            st.markdown("<br>", unsafe_allow_html=True)
            fig_up_traj = px.line(
                unit_up_preds,
                x="cycle",
                y=["ANN_prediction", "CNN_prediction", "LSTM_prediction", "CNNLSTM_prediction", "ensemble_prediction"],
                labels={"value": "Predicted RUL (cycles)", "cycle": "Operating Cycle", "variable": "Model Architecture"},
                title=f"Uploaded Engine #{cur_up_unit} — All Model Predictions Across Recorded Life",
            )
            apply_aerospace_theme(fig_up_traj, height=400)
            st.plotly_chart(fig_up_traj, use_container_width=True)

            # Download Predictions Button
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=up_df.to_csv(index=False).encode("utf-8"),
                file_name=f"aeropulse_predictions_{uploaded_file.name if uploaded_file else 'uploaded'}",
                mime="text/csv",
                use_container_width=True,
            )


# =============================================================================
# PAGE 5: MODEL COMPARISON
# =============================================================================
elif st.session_state.nav_page == "⚖️ Model Comparison":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">MODEL ARCHITECTURE & <span>BENCHMARK COMPARISON</span></h1>
            <div class="aero-subtitle">Verified Evaluation Across ANN, CNN, LSTM, and CNN-LSTM on the Held-out NASA C-MAPSS Test Partition</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Offline Test Benchmark Metrics (Student 2 & Student 3 Results)")
        st.dataframe(
            model_metrics_df.style.format({
                "MAE": "{:.4f} cycles",
                "RMSE": "{:.4f} cycles",
                "R2": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_mc1, col_mc2, col_mc3 = st.columns(3)

    with col_mc1:
        with st.container(border=True):
            fig_mae = px.bar(
                model_metrics_df,
                x="Model",
                y="MAE",
                color="Model",
                color_discrete_sequence=["#28C5F5", "#8B5CF6", "#24C78E", "#EC4899"],
                title="Mean Absolute Error (Lower is Better)",
                text=model_metrics_df["MAE"].apply(lambda v: f"{v:.2f}"),
            )
            fig_mae.update_traces(textposition="outside")
            apply_aerospace_theme(fig_mae, height=330)
            st.plotly_chart(fig_mae, use_container_width=True)

    with col_mc2:
        with st.container(border=True):
            fig_rmse = px.bar(
                model_metrics_df,
                x="Model",
                y="RMSE",
                color="Model",
                color_discrete_sequence=["#28C5F5", "#8B5CF6", "#24C78E", "#EC4899"],
                title="Root Mean Squared Error (Lower is Better)",
                text=model_metrics_df["RMSE"].apply(lambda v: f"{v:.2f}"),
            )
            fig_rmse.update_traces(textposition="outside")
            apply_aerospace_theme(fig_rmse, height=330)
            st.plotly_chart(fig_rmse, use_container_width=True)

    with col_mc3:
        with st.container(border=True):
            fig_r2 = px.bar(
                model_metrics_df,
                x="Model",
                y="R2",
                color="Model",
                color_discrete_sequence=["#28C5F5", "#8B5CF6", "#24C78E", "#EC4899"],
                title="R² Coefficient of Determination (Higher is Better)",
                text=model_metrics_df["R2"].apply(lambda v: f"{v:.3f}"),
            )
            fig_r2.update_traces(textposition="outside")
            apply_aerospace_theme(fig_r2, height=330)
            st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Architectural Comparison & Input Shapes")
        arch_df = pd.DataFrame({
            "Architecture": ["ANN (Student 2)", "CNN (Student 2)", "LSTM (Student 3)", "CNN-LSTM (Student 3)"],
            "Input Shape": ["(17,)", "(17, 1)", "(30, 17)", "(30, 17)"],
            "Temporal Window": ["Instantaneous (1 cycle)", "Instantaneous (1 cycle)", "30 Consecutive Cycles", "30 Consecutive Cycles"],
            "Core Layers": ["Dense (128 -> 64 -> 32)", "Conv1D (64, 128) + MaxPool", "LSTM (64 -> 32)", "Conv1D (64, 128) + LSTM (64)"],
            "Key Trade-off": [
                "Fastest inference; predicts from Cycle 1; lacks temporal memory",
                "Extracts cross-sensor local spatial features; lacks sequence memory",
                "Captures long-term degradation trajectories; requires 30-cycle history",
                "Extracts spatial features + temporal sequence memory; achieves best overall R² (0.614)",
            ],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 6: SENSOR ANALYTICS
# =============================================================================
elif st.session_state.nav_page == "📡 Sensor Analytics":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">SENSOR TELEMETRY & <span>DEGRADATION ANALYTICS</span></h1>
            <div class="aero-subtitle">Investigate Telemetry Channels Correlated with Engine Wear and Failure Dynamics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sa1, col_sa2 = st.columns([1, 1])

    with col_sa1:
        with st.container(border=True):
            st.subheader("Top Sensors Correlated with Remaining Useful Life")
            st.caption("Pearson correlation magnitude |r| with true RUL across training run-to-failure cycles.")

            fig_imp = px.bar(
                sensor_importance_df.head(10).sort_values("importance_score", ascending=True),
                x="importance_score",
                y="sensor",
                orientation="h",
                color="importance_score",
                color_continuous_scale="Blues",
                title="Top 10 Most Influential Telemetry Channels",
            )
            apply_aerospace_theme(fig_imp, height=380)
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_sa2:
        with st.container(border=True):
            st.subheader("Multi-Sensor Overlay for Selected Engine")
            unit_trajectory, min_c, max_c = render_engine_cycle_toolbar(show_dataset_toggle=False)
            s_unit = st.session_state.selected_unit

            s_data = c_mapss_raw_df[c_mapss_raw_df["unit"] == s_unit].sort_values("cycle")

            chosen_sensors = st.multiselect(
                "Select Sensor Channels to Overlay:",
                FEATURES,
                default=["sensor_11", "sensor_4", "sensor_12"],
            )

            if chosen_sensors:
                fig_multi_s = px.line(
                    s_data,
                    x="cycle",
                    y=chosen_sensors,
                    title=f"Engine #{s_unit} — Telemetry Dynamics",
                )
                apply_aerospace_theme(fig_multi_s, height=330)
                st.plotly_chart(fig_multi_s, use_container_width=True)


# =============================================================================
# PAGE 7: RESEARCH CONTRIBUTION
# =============================================================================
elif st.session_state.nav_page == "🔬 Research Contribution":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">RESEARCH CONTRIBUTION — <span>MODEL DISAGREEMENT VS PREDICTION ERROR</span></h1>
            <div class="aero-subtitle">Investigating whether multi-architecture neural network disagreement provides an early uncertainty signal.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            ### Research Hypothesis Formulation
            > **Hypothesis:** When multiple distinct neural-network architectures (point dense ANN, 1D-convolutional CNN, recurrent LSTM, and hybrid CNN-LSTM) produce divergent RUL predictions on the same engine cycle, that cycle corresponds to non-standard sensor drift, atypical degradation rates, or heightened prediction difficulty—manifesting as higher actual prediction error.

            We analyze all **1,961 aligned test evaluations** from the held-out NASA C-MAPSS test engines where run-to-failure ground truth is verified.
            """
        )

    # Statistical Correlations
    pearson_r = combined_test_df["disagreement_score"].corr(combined_test_df["ensemble_abs_error"], method="pearson")
    spearman_rho = combined_test_df["disagreement_score"].corr(combined_test_df["ensemble_abs_error"], method="spearman")

    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(render_kpi_card("PEARSON CORRELATION (r)", f"{pearson_r:.4f}", "Linear relationship (p < 0.001)"), unsafe_allow_html=True)
    with sc2:
        st.markdown(render_kpi_card("SPEARMAN RANK CORRELATION (ρ)", f"{spearman_rho:.4f}", "Monotonic relationship (p < 0.001)"), unsafe_allow_html=True)
    with sc3:
        st.markdown(render_kpi_card("ALIGNED TEST SAMPLES", f"{len(combined_test_df):,}", "Held-out C-MAPSS FD001 Partition"), unsafe_allow_html=True)

    # Scatter Plot with Pure NumPy Regression Trendline
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        fig_scatter = px.scatter(
            combined_test_df,
            x="disagreement_score",
            y="ensemble_abs_error",
            color="health_status",
            color_discrete_map={"Healthy": "#24C78E", "Warning": "#F5B942", "Critical": "#F06472"},
            opacity=0.45,
            labels={
                "disagreement_score": "Model Disagreement (Standard Deviation across 4 Models in Cycles)",
                "ensemble_abs_error": "Actual Absolute Error of Ensemble Prediction (Cycles)",
                "health_status": "Engine Health Status",
            },
            title=f"Experimental Scatter Plot: Model Disagreement vs Absolute Error (N = {len(combined_test_df):,})",
        )

        # Pure NumPy OLS Trendline
        x_vals = combined_test_df["disagreement_score"].values
        y_vals = combined_test_df["ensemble_abs_error"].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
        y_line = slope * x_line + intercept

        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Trendline (Slope = +{slope:.2f} cycles error / cycle disagreement)",
                line=dict(color="#FFFFFF", width=2.5, dash="dash"),
            )
        )
        apply_aerospace_theme(fig_scatter, height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Grouped Error Analysis across Terciles
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Error Distribution Across Empirical Disagreement Tiers")

        combined_test_df["Tier"] = pd.cut(
            combined_test_df["disagreement_score"],
            bins=[-np.inf, DISAGREEMENT_LOW, DISAGREEMENT_HIGH, np.inf],
            labels=["Low Disagreement (<= 7.41)", "Moderate Disagreement (7.41 - 15.04)", "High Disagreement (> 15.04)"],
        )

        tier_summary = combined_test_df.groupby("Tier", observed=True)["ensemble_abs_error"].agg(
            Count="count",
            Mean_Absolute_Error="mean",
            Median_Absolute_Error="median",
            Std_Dev="std",
        ).reset_index()

        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.dataframe(
                tier_summary.style.format({
                    "Mean_Absolute_Error": "{:.2f} cycles",
                    "Median_Absolute_Error": "{:.2f} cycles",
                    "Std_Dev": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with col_t2:
            fig_box = px.box(
                combined_test_df,
                x="Tier",
                y="ensemble_abs_error",
                color="Tier",
                color_discrete_sequence=["#28C5F5", "#F5B942", "#F06472"],
                title="Prediction Error Spread by Disagreement Tier",
                labels={"ensemble_abs_error": "Absolute Prediction Error (cycles)"},
            )
            apply_aerospace_theme(fig_box, height=360)
            st.plotly_chart(fig_box, use_container_width=True)

    # Scientific Conclusion & Viva Points
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <h4 style="color:#28C5F5; margin-top:0;">Scientific Summary & Viva Defense Points</h4>
            <ul style="color:#F4F7FC; line-height:1.7; font-size:0.95rem;">
                <li><b>Support for the Hypothesis:</b> The empirical results demonstrate that observations in the <b>High Disagreement</b> group exhibit a mean prediction error of <b>39.72 cycles</b> (median 30.89 cycles), compared to only <b>18.10 cycles</b> (median 7.36 cycles) in the <b>Low Disagreement</b> group—a <b>2.19× increase in actual prediction error</b>.</li>
                <li><b>Correlation Strength:</b> A statistically significant positive correlation exists (Pearson r = <b>0.2565</b>, Spearman ρ = <b>0.3631</b>, both p < 10<sup>-20</sup>). While disagreement does not guarantee an error in every single instance, it provides a meaningful statistical signal.</li>
                <li><b>Aviation Safety Application:</b> In real-world operational deployments where run-to-failure ground truth is unknown during flight, <b>model disagreement serves as an unsupervised confidence metric</b>. When disagreement exceeds 15 cycles, maintenance personnel are alerted to prioritize secondary inspection.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 8: ABOUT PROJECT
# =============================================================================
elif st.session_state.nav_page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="aero-header">
            <h1 class="aero-title">AEROPULSE AI — <span>SYSTEM SPECIFICATIONS & TEAM CONTRIBUTIONS</span></h1>
            <div class="aero-subtitle">AI-Based Jet Engine Health Monitoring & Remaining Useful Life Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Project Architecture Overview")
        st.markdown(
            """
            AeroPulse AI is an advanced predictive maintenance intelligence platform developed using the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset.
            The system integrates classical dense neural networks (ANN), 1D convolutional feature extractors (CNN), recurrent sequence networks (LSTM), and hybrid spatio-temporal architectures (CNN-LSTM) into an ensemble diagnostic dashboard.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_ab1, col_ab2 = st.columns(2)

    with col_ab1:
        with st.container(border=True):
            st.markdown(
                """
                <h4 style="color:#24C78E; margin-top:0;">Student 1 — Data Preprocessing & Pipeline</h4>
                <ul style="color:#B7C5D8; font-size:0.9rem; line-height:1.6;">
                    <li>Dataset ingestion: NASA C-MAPSS FD001.</li>
                    <li>Feature selection: Screened 21 sensors down to 17 informative channels; dropped 7 flatline sensors.</li>
                    <li>Piecewise RUL calculation & initial 80/20 engine split.</li>
                </ul>
                <h4 style="color:#28C5F5; margin-top:16px;">Student 2 — Point Neural Networks (ANN & CNN)</h4>
                <ul style="color:#B7C5D8; font-size:0.9rem; line-height:1.6;">
                    <li>Engine-wise 80/10/10 split (engines 1-80 train, 81-90 val, 91-100 test).</li>
                    <li>Fitted training StandardScaler on the 17 features.</li>
                    <li>Built & trained ANN (Dense 128-64-32) and CNN (Conv1D-MaxPool-Dense).</li>
                    <li>MAE: ANN = 37.69 cycles | CNN = 39.51 cycles.</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    with col_ab2:
        with st.container(border=True):
            st.markdown(
                """
                <h4 style="color:#8B5CF6; margin-top:0;">Student 3 — Temporal Sequence Networks (LSTM & CNN-LSTM)</h4>
                <ul style="color:#B7C5D8; font-size:0.9rem; line-height:1.6;">
                    <li>Built 30-cycle sliding window sequence generator strictly respecting engine boundaries.</li>
                    <li>Trained LSTM (64-32 recurrent cells) and CNN-LSTM hybrid architecture.</li>
                    <li>Achieved significant performance improvement: LSTM MAE = 31.16 | CNN-LSTM MAE = 31.10, R² = 0.614.</li>
                </ul>
                <h4 style="color:#F5B942; margin-top:16px;">Student 4 / Final Integration — Intelligence Dashboard</h4>
                <ul style="color:#B7C5D8; font-size:0.9rem; line-height:1.6;">
                    <li>Unified all four models into an interactive mission-control dashboard.</li>
                    <li>Implemented 3 User Input Levels: Engine selection, Cycle slider, and CSV Upload.</li>
                    <li>Conducted the core research contribution investigating Model Disagreement vs Prediction Error.</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Technical Stack")
        st.markdown(
            """
            - **Framework:** Streamlit 1.54
            - **Deep Learning:** TensorFlow / Keras 2.21 (ANN, CNN, LSTM, CNN-LSTM)
            - **Data & Feature Engineering:** Scikit-Learn 1.7, Pandas 2.3, NumPy 1.24
            - **Visualizations:** Plotly Express & Plotly Graph Objects (Interactive Dark Aerospace Theme)
            - **Runtime Environment:** Python 3.13 (Windows Compatible)
            """
        )
