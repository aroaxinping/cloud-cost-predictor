"""Shared constants, data loaders, and helper components."""
from functools import wraps
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "clean"
MODELS_DIR = ROOT / "models"

GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
GRAY = "#94A3B8"
DARK_BG = "#000000"
CARD_BG = "#0A1628"
TEXT = "#FAFAFA"
TEXT_DIM = "#94A3B8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #0A1628;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #FAFAFA;
        font-size: 1.05rem;
        padding: 0.5rem 0;
    }

    .kpi-card {
        background: linear-gradient(135deg, #0A1628 0%, #0F1D33 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #0F1D33;
        transition: transform 0.2s ease;
        margin-bottom: 1rem;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8B949E;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-context {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }

    .story-text {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #C9D1D9;
        max-width: 720px;
        margin: 1rem 0 2rem 0;
    }
    .story-text strong {
        color: #FAFAFA;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 3.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4B5563;
        display: inline-block;
    }

    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6B7280;
        font-size: 0.8rem;
        border-top: 1px solid #0A1628;
        margin-top: 3rem;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


class DataNotReady(Exception):
    pass


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise DataNotReady(
            f"**{path.name}** not found.  \n"
            "Run `make all` (or the individual pipeline steps) to generate the data files."
        )
    return path


def safe_page(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DataNotReady as e:
            st.error(str(e))
            st.stop()
    return wrapper


@st.cache_data
def load_utilization():
    return pd.read_csv(_check_file(DATA / "vm_utilization_summary.csv"))


@st.cache_data
def load_classified():
    return pd.read_csv(_check_file(DATA / "vm_classified.csv"))


@st.cache_data
def load_recommendations():
    return pd.read_csv(_check_file(DATA / "vm_recommendations.csv"))


@st.cache_data
def load_fleet_costs():
    return pd.read_csv(_check_file(DATA / "fleet_cost_estimate.csv"))


@st.cache_data
def load_models():
    models = {}
    for q in [0.10, 0.50, 0.95]:
        path = _check_file(MODELS_DIR / f"quantile_{q:.2f}.json")
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models


def kpi_card(value, label, context="", color=GREEN):
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value" style="color: {color};">{value}</p>
        <p class="kpi-label">{label}</p>
        <p class="kpi-context">{context}</p>
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown("""
    <div class="app-footer">
        Built by Aroa &nbsp;|&nbsp;
        <a href="https://github.com/aroaxinping" target="_blank"
           style="color: #C9D1D9; text-decoration: none;">GitHub</a> &nbsp;|&nbsp;
        Data: SAP Cloud Infrastructure Dataset (CC BY 4.0)
    </div>
    """, unsafe_allow_html=True)
