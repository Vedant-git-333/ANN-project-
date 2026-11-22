"""
AeroPulse AI — Aerospace Mission Control Theme & UI Components
High-Contrast Redesign adhering to strict palette specifications:
- Main Background: Deep Navy #08111F
- Sidebar Background: #0B1524
- Cards & Containers: #111F33
- Primary Text: #F4F7FC
- Secondary Text: #B7C5D8
- Accent Cyan: #28C5F5
- Success: #24C78E
- Warning: #F5B942
- Error / Critical: #F06472
"""

import streamlit as st
import plotly.graph_objects as go


def apply_custom_css():
    """Injects high-contrast aerospace mission-control styling into Streamlit."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

    /* 1. Global App Background & Base Typography */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #08111F !important;
        color: #F4F7FC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(8, 17, 31, 0.85) !important;
    }

    /* 2. Sidebar Redesign */
    section[data-testid="stSidebar"] {
        background-color: #0B1524 !important;
        border-right: 1px solid #1E314B !important;
    }

    section[data-testid="stSidebar"] * {
        color: #F4F7FC !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #B7C5D8 !important;
    }

    /* Sidebar Radio Navigation items */
    div[data-testid="stRadio"] > label {
        color: #28C5F5 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #111F33 !important;
        border: 1px solid #1E314B !important;
        border-radius: 6px !important;
        padding: 8px 14px !important;
        margin-bottom: 6px !important;
        color: #F4F7FC !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #28C5F5 !important;
        background: #162942 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-color: #28C5F5 !important;
        background: rgba(40, 197, 245, 0.15) !important;
        color: #28C5F5 !important;
        font-weight: 700 !important;
    }

    /* 3. Streamlit Native Containers (Bordered) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111F33 !important;
        border: 1px solid #1E314B !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* 4. Headings & Markdown High Contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #F4F7FC !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }

    h1 { font-size: 2.1rem !important; }
    h2 { font-size: 1.6rem !important; margin-top: 0.5rem !important; }
    h3 { font-size: 1.25rem !important; }

    p, span, label, div {
        color: #F4F7FC;
    }

    .stMarkdown p, .stCaption {
        color: #B7C5D8 !important;
        font-size: 0.92rem !important;
    }

    /* 5. Form Widgets & Input Contrast */
    /* Widget Labels */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #F4F7FC !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] {
        background-color: #111F33 !important;
        border: 1px solid #1E314B !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] * {
        color: #F4F7FC !important;
        background-color: transparent !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #111F33 !important;
        border: 1px solid #28C5F5 !important;
    }

    li[data-baseweb="menu-item"] {
        color: #F4F7FC !important;
        background-color: #111F33 !important;
    }

    li[data-baseweb="menu-item"]:hover {
        background-color: #1A2E4C !important;
        color: #28C5F5 !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #F4F7FC !important;
    }

    div[data-testid="stSlider"] label {
        color: #F4F7FC !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #28C5F5 !important;
        color: #08111F !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        background-color: #58D3F7 !important;
        box-shadow: 0 0 12px rgba(40, 197, 245, 0.5) !important;
    }

    button[kind="secondary"] {
        background-color: #111F33 !important;
        border: 1px solid #28C5F5 !important;
        color: #F4F7FC !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #162942 !important;
        border-color: #58D3F7 !important;
        color: #28C5F5 !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #111F33 !important;
        border: 1px solid #1E314B !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #B7C5D8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #28C5F5 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 1px solid #1E314B !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111F33 !important;
        border: 1px solid #1E314B !important;
        border-bottom: none !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 18px !important;
        color: #B7C5D8 !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(40, 197, 245, 0.15) !important;
        color: #28C5F5 !important;
        border-top: 2px solid #28C5F5 !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: #111F33 !important;
        border: 2px dashed #28C5F5 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: #F4F7FC !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E314B !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* 6. Custom Header */
    .aero-header {
        background: linear-gradient(90deg, #111F33 0%, #0E1A2B 100%);
        border: 1px solid #1E314B;
        border-left: 5px solid #28C5F5;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    }

    .aero-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #F4F7FC;
        margin: 0;
        text-transform: uppercase;
    }

    .aero-title span {
        color: #28C5F5;
    }

    .aero-subtitle {
        color: #B7C5D8;
        font-size: 0.95rem;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }

    /* 7. High-Contrast KPI Cards */
    .kpi-card {
        background: #111F33;
        border: 1px solid #1E314B;
        border-radius: 8px;
        padding: 16px;
        transition: all 0.2s ease;
        height: 100%;
        box-sizing: border-box;
    }

    .kpi-card:hover {
        border-color: #28C5F5;
        box-shadow: 0 4px 16px rgba(40, 197, 245, 0.15);
    }

    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #B7C5D8;
        margin-bottom: 4px;
        font-weight: 600;
    }

    .kpi-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #F4F7FC;
        line-height: 1.2;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #B7C5D8;
        margin-top: 6px;
    }

    /* 8. Status Badges */
    .badge-healthy {
        display: inline-block;
        background: rgba(36, 199, 142, 0.15);
        color: #24C78E;
        border: 1px solid #24C78E;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }

    .badge-warning {
        display: inline-block;
        background: rgba(245, 185, 66, 0.15);
        color: #F5B942;
        border: 1px solid #F5B942;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }

    .badge-critical {
        display: inline-block;
        background: rgba(240, 100, 114, 0.18);
        color: #F06472;
        border: 1px solid #F06472;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }

    .badge-agreement-high {
        display: inline-block;
        background: rgba(36, 199, 142, 0.15);
        color: #24C78E;
        border: 1px solid #24C78E;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
    }

    .badge-agreement-mod {
        display: inline-block;
        background: rgba(245, 185, 66, 0.15);
        color: #F5B942;
        border: 1px solid #F5B942;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
    }

    .badge-agreement-disagree {
        display: inline-block;
        background: rgba(240, 100, 114, 0.2);
        color: #F06472;
        border: 1px solid #F06472;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.82rem;
    }

    /* 9. Banners */
    .banner-success {
        background: rgba(36, 199, 142, 0.12);
        border-left: 4px solid #24C78E;
        border-radius: 6px;
        padding: 12px 18px;
        margin: 12px 0;
        color: #F4F7FC;
        font-size: 0.92rem;
    }

    .banner-warning {
        background: rgba(245, 185, 66, 0.12);
        border-left: 4px solid #F5B942;
        border-radius: 6px;
        padding: 12px 18px;
        margin: 12px 0;
        color: #F4F7FC;
        font-size: 0.92rem;
    }

    .banner-danger {
        background: rgba(240, 100, 114, 0.15);
        border-left: 4px solid #F06472;
        border-radius: 6px;
        padding: 12px 18px;
        margin: 12px 0;
        color: #F4F7FC;
        font-size: 0.92rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def apply_aerospace_theme(fig, height=400):
    """Applies unified dark aerospace mission-control styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(8, 17, 31, 0)",
        plot_bgcolor="rgba(17, 31, 51, 0.55)",
        font=dict(family="Inter, sans-serif", size=12, color="#B7C5D8"),
        title=dict(
            font=dict(family="Rajdhani, sans-serif", size=17, color="#F4F7FC"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=45, r=25, t=45, b=40),
        legend=dict(
            bgcolor="rgba(17, 31, 51, 0.85)",
            bordercolor="#1E314B",
            borderwidth=1,
            font=dict(size=11, color="#F4F7FC"),
        ),
        xaxis=dict(
            gridcolor="#1E314B",
            zerolinecolor="#28C5F5",
            linecolor="#1E314B",
            tickfont=dict(color="#B7C5D8", size=11),
            title=dict(font=dict(color="#F4F7FC", size=12)),
        ),
        yaxis=dict(
            gridcolor="#1E314B",
            zerolinecolor="#28C5F5",
            linecolor="#1E314B",
            tickfont=dict(color="#B7C5D8", size=11),
            title=dict(font=dict(color="#F4F7FC", size=12)),
        ),
        hoverlabel=dict(
            bgcolor="#111F33",
            bordercolor="#28C5F5",
            font=dict(color="#F4F7FC", family="Inter, sans-serif"),
        ),
    )
    return fig


def render_kpi_card(label, value, subtext="", badge_html=""):
    """Renders a self-contained, high-contrast HTML KPI card."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext} {badge_html}</div>
    </div>
    """
    return html
