"""Overview page — front door of the ProcessRCA dashboard."""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_loader import load_eval_auc, load_causal_summary, load_training_log


def render() -> None:
    st.title("ProcessRCA")
    st.markdown(
        "**Causal Root Cause Analysis for Multivariate Industrial Time-Series**"
    )
    st.markdown(
        "A digital-twin extension that combines deep-learning anomaly detection "
        "with time-series causal discovery to identify *which* sensor drove a "
        "process deviation — not just *that* a deviation occurred."
    )

    st.markdown("---")

    # --- Quick facts (top row: scope) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Process variables", "52")
    col2.metric("Fault types", "20")
    col3.metric("Anomaly windows", "92")
    col4.metric("Total timesteps", "15.3M")

    # --- Headline results (second row: outcomes) ---
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Median detection AUC", "0.91", help="LSTM autoencoder, ROC-AUC across 20 faults")
    col6.metric("Strong-detection faults", "17 / 20", help="ROC-AUC ≥ 0.85")
    col7.metric("Top-3 RCA hit rate", "0.29", help="Across 92 anomaly windows; random baseline = 0.058 (3/52)")
    col8.metric("Faults with ≥80% top-3", "8 / 20", help="Faults 1, 10, 11, 12, 14, 17 reach 80–100%")

    st.markdown("")

    # --- Pipeline diagram (HTML for cleaner look) ---
    st.subheader("Pipeline")
    pipeline_html = """
    <div style="display:flex; flex-direction:column; gap:8px; margin:12px 0 24px 0;">

      <div style="display:flex; align-items:center; gap:14px;">
        <div style="flex:0 0 220px; padding:10px 14px; background:#1f3a5f; color:white;
                    border-radius:6px; font-weight:600;">Stage 1 · Detection</div>
        <div style="color:#3d4f5d; font-size:14px;">
          LSTM autoencoder trained on normal-operation windows only<br>
          <span style="color:#7a8593;">→ per-window reconstruction error</span>
        </div>
      </div>

      <div style="margin-left:90px; color:#9aa5b1;">▼</div>

      <div style="display:flex; align-items:center; gap:14px;">
        <div style="flex:0 0 220px; padding:10px 14px; background:#2b5d6e; color:white;
                    border-radius:6px; font-weight:600;">Stage 2 · Localization</div>
        <div style="color:#3d4f5d; font-size:14px;">
          Threshold + persistence rule (10 consecutive timesteps above 99th-percentile)<br>
          <span style="color:#7a8593;">→ deviation onset timestamp</span>
        </div>
      </div>

      <div style="margin-left:90px; color:#9aa5b1;">▼</div>

      <div style="display:flex; align-items:center; gap:14px;">
        <div style="flex:0 0 220px; padding:10px 14px; background:#2b7a78; color:white;
                    border-radius:6px; font-weight:600;">Stage 3 · Causal discovery</div>
        <div style="color:#3d4f5d; font-size:14px;">
          PCMCI with ParCorr independence test, τ_max = 5<br>
          <span style="color:#7a8593;">→ time-lagged causal graph (52×52×6)</span>
        </div>
      </div>

      <div style="margin-left:90px; color:#9aa5b1;">▼</div>

      <div style="display:flex; align-items:center; gap:14px;">
        <div style="flex:0 0 220px; padding:10px 14px; background:#c44536; color:white;
                    border-radius:6px; font-weight:600;">Stage 4 · Ranking</div>
        <div style="color:#3d4f5d; font-size:14px;">
          Hub-aware: causal strength × deviation magnitude − hub penalty<br>
          <span style="color:#7a8593;">→ top-K root-cause candidates</span>
        </div>
      </div>

    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.markdown("---")

    # --- Detailed results breakdown ---
    st.subheader("Results detail")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("**Detection (LSTM autoencoder)**")
        auc = load_eval_auc()
        median_auc = float(auc["roc_auc"].median())
        mean_auc = float(auc["roc_auc"].mean())
        n_strong = int((auc["roc_auc"] >= 0.85).sum())
        st.markdown(
            f"- Median ROC-AUC: **{median_auc:.2f}**\n"
            f"- Mean ROC-AUC: **{mean_auc:.2f}**\n"
            f"- Faults with AUC ≥ 0.85: **{n_strong} / 20**\n"
            f"- Weak (incipient) faults: 3, 9, 15 — known hard in literature"
        )

    with res_col2:
        st.markdown("**Causal RCA (top-3 hit rate over 92 windows)**")
        st.markdown(
            f"- v1 baseline (count-based): **0.141**\n"
            f"- v2 (+deviation filter): **0.250**\n"
            f"- v3 (+hub penalty, lags 1–3): **0.293**\n"
            f"- Random baseline (3/52 vars): 0.058"
        )

    st.markdown("")

    # --- Per-fault top-3 bar chart ---
    st.subheader("Per-fault top-3 hit rate")
    causal = load_causal_summary("v3")
    causal = causal[causal["fault_number"] != "OVERALL"].copy()
    causal["fault_number"] = causal["fault_number"].astype(int)
    causal["top3_acc"] = causal["top3_acc"].astype(float)
    causal = causal.sort_values("fault_number")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"f{n:02d}" for n in causal["fault_number"]],
        y=causal["top3_acc"],
        marker_color=[
            "#2b7a78" if v >= 0.6 else
            "#c4a35a" if v >= 0.2 else
            "#a85a5a"
            for v in causal["top3_acc"]
        ],
        text=[f"{v:.0%}" for v in causal["top3_acc"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=380,
        xaxis_title="Fault type",
        yaxis_title="Top-3 hit rate",
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Green = strong (≥60%), amber = partial (≥20%), red = miss. "
        "Faults 3, 9, 15 are incipient (subtle) and produce few detectable windows; "
        "their causal accuracy is bounded by detection coverage upstream."
    )

    st.markdown("---")

    # --- Training curve ---
    st.subheader("Training curve (LSTM autoencoder)")
    log = load_training_log()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=log["epoch"], y=log["train_mse"], mode="lines",
                              name="train MSE", line=dict(width=1.8)))
    fig2.add_trace(go.Scatter(x=log["epoch"], y=log["val_mse"], mode="lines",
                              name="val MSE", line=dict(width=1.8)))
    fig2.update_layout(
        height=320, xaxis_title="Epoch", yaxis_title="MSE (scaled)",
        legend=dict(x=0.7, y=0.95), margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Use the sidebar to navigate to **Detection** for per-fault ROC analysis "
        "and **Causal RCA** to explore individual anomaly windows with their "
        "predicted root causes."
    )