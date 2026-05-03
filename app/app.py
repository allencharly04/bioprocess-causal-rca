"""ProcessRCA — main Streamlit app entry point."""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

# Make local app modules importable regardless of where streamlit is launched from
THIS = Path(__file__).resolve().parent
sys.path.append(str(THIS))

import pages_overview
import pages_detection
import pages_rca

st.set_page_config(
    page_title="ProcessRCA — Causal Root Cause Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
st.sidebar.title("ProcessRCA")
st.sidebar.caption("Causal RCA for industrial time-series")
st.sidebar.markdown("---")

PAGES = {
    "Overview": pages_overview.render,
    "Fault Detection": pages_detection.render,
    "Causal RCA": pages_rca.render,
}

choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack**\n\n"
    "- PyTorch (LSTM autoencoder)\n"
    "- tigramite (PCMCI)\n"
    "- Streamlit + Plotly\n"
    "- scikit-learn"
)
st.sidebar.markdown(
    "**Benchmark**\n\n"
    "Tennessee Eastman Process (Rieth et al. 2017)"
)

# --- Render selected page ---
PAGES[choice]()