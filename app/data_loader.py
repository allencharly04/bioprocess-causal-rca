"""Cached data loaders for the Streamlit dashboard."""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from windowing import PROCESS_COLS, apply_scaler, load_scaler  # noqa: E402
from model import LSTMAutoencoder  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


@st.cache_data(show_spinner=False)
def load_eval_auc() -> pd.DataFrame:
    return pd.read_csv(MODELS / "eval_auc.csv")


@st.cache_data(show_spinner=False)
def load_causal_summary(version: str = "v3") -> pd.DataFrame:
    return pd.read_csv(MODELS / f"causal_summary_{version}.csv")


@st.cache_data(show_spinner=False)
def load_eval_scores() -> dict:
    """{'normal': arr, 'fault_01': arr, ..., 'fault_20': arr}"""
    data = np.load(MODELS / "eval_scores.npz")
    return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def load_causal_results(version: str = "v3") -> dict:
    fname = f"causal_results_{version}.npz"
    raw = np.load(MODELS / fname, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


@st.cache_data(show_spinner=False)
def load_anomaly_windows() -> dict:
    raw = np.load(PROC / "anomaly_windows.npz", allow_pickle=True)
    return {k: raw[k] for k in raw.files}


@st.cache_data(show_spinner=False)
def load_training_log() -> pd.DataFrame:
    return pd.read_csv(MODELS / "training_log.csv")


@st.cache_data(show_spinner=False)
def get_process_cols() -> list[str]:
    return list(PROCESS_COLS)


@st.cache_data(show_spinner=False)
def load_faulty_run(fault_n: int, run_id: int) -> pd.DataFrame | None:
    """Load a single (fault, run) from the faulty test parquet, scaled."""
    df = pd.read_parquet(PROC / "TEP_Faulty_Testing.parquet", engine="fastparquet")
    sub = df[(df["faultNumber"] == fault_n) & (df["simulationRun"] == run_id)].sort_values("sample")
    if len(sub) == 0:
        return None
    return sub.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LSTMAutoencoder(
        n_features=52, hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"], window_size=50, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def get_scaler():
    return load_scaler(PROC / "scaler.joblib")


# Fault descriptions for display
FAULT_DESCRIPTIONS: dict[int, str] = {
    1: "A/C feed ratio step (Stream 4)",
    2: "B composition step (Stream 4)",
    3: "D feed temperature (Stream 2)",
    4: "Reactor cooling water inlet temperature step",
    5: "Condenser cooling water inlet temperature step",
    6: "A feed loss (Stream 1)",
    7: "C header pressure loss (Stream 4)",
    8: "A, B, C feed composition random",
    9: "D feed temperature random",
    10: "C feed temperature random",
    11: "Reactor cooling water inlet temp random",
    12: "Condenser cooling water inlet temp random",
    13: "Reaction kinetics slow drift",
    14: "Reactor cooling water valve sticking",
    15: "Condenser cooling water valve sticking",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
}