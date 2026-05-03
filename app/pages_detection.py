"""Detection page — explore per-fault detection performance interactively."""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_curve

from data_loader import (
    load_eval_auc, load_eval_scores, load_faulty_run,
    load_autoencoder, get_scaler, get_process_cols, FAULT_DESCRIPTIONS,
)


def _per_timestep_error(model, run_array_scaled: np.ndarray, device) -> np.ndarray:
    """Slide window=50 across one run and average overlapping per-timestep errors."""
    T = run_array_scaled.shape[0]
    if T < 50:
        return np.zeros(T, dtype=np.float32)
    starts = np.arange(0, T - 50 + 1, 1)
    sub = np.stack([run_array_scaled[s:s + 50] for s in starts], axis=0).astype(np.float32)
    err_acc = np.zeros(T, dtype=np.float64)
    cnt_acc = np.zeros(T, dtype=np.int64)
    bs = 64
    with torch.no_grad():
        for i in range(0, len(sub), bs):
            xb = torch.from_numpy(sub[i:i + bs]).to(device)
            ts_err = model.reconstruction_error(xb, "per_timestep").cpu().numpy()
            for k in range(ts_err.shape[0]):
                s = starts[i + k]
                err_acc[s:s + 50] += ts_err[k]
                cnt_acc[s:s + 50] += 1
    cnt_acc[cnt_acc == 0] = 1
    return (err_acc / cnt_acc).astype(np.float32)


def render() -> None:
    st.title("Fault Detection")
    st.markdown(
        "The LSTM autoencoder is trained on normal-operation windows only. "
        "Reconstruction error becomes the deviation score: high error = anomaly. "
        "Pick a fault below to explore detection performance."
    )

    auc_df = load_eval_auc()
    fault_options = sorted(auc_df["fault_number"].astype(int).tolist())

    # --- Fault selector ---
    col_a, col_b = st.columns([1, 3])
    with col_a:
        fault_n = st.selectbox(
            "Fault type", fault_options,
            format_func=lambda f: f"Fault {int(f):02d}",
        )
    with col_b:
        st.markdown(
            f"**Fault {int(fault_n):02d}**: {FAULT_DESCRIPTIONS.get(int(fault_n), '?')}"
        )
        this_auc = float(auc_df.loc[auc_df["fault_number"] == fault_n, "roc_auc"].iloc[0])
        n_w = int(auc_df.loc[auc_df["fault_number"] == fault_n, "n_windows"].iloc[0])
        st.markdown(f"**ROC-AUC**: {this_auc:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Test windows**: {n_w:,}")

    st.markdown("---")

    # --- Score distribution + ROC side by side ---
    scores = load_eval_scores()
    s_normal = scores["normal"]
    s_fault = scores[f"fault_{int(fault_n):02d}"]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Score distribution")
        fig_dist = go.Figure()
        # Clip to the 99.5th percentile for sane axes
        cap = float(np.percentile(np.concatenate([s_normal, s_fault]), 99.5))
        bins = np.linspace(0, cap, 60)
        fig_dist.add_trace(go.Histogram(
            x=np.clip(s_normal, 0, cap), xbins=dict(start=0, end=cap, size=cap / 60),
            name="normal", marker_color="#2b7a78", opacity=0.65, histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=np.clip(s_fault, 0, cap), xbins=dict(start=0, end=cap, size=cap / 60),
            name=f"fault {int(fault_n)}", marker_color="#c44536", opacity=0.65,
            histnorm="probability density",
        ))
        fig_dist.update_layout(
            barmode="overlay", height=340,
            xaxis_title="reconstruction error",
            yaxis_title="density",
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("ROC curve")
        y_true = np.concatenate([np.zeros(len(s_normal)), np.ones(len(s_fault))])
        y_score = np.concatenate([s_normal, s_fault])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color="#1f3a5f", width=2),
                                     name=f"AUC = {this_auc:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="#999", dash="dash"),
                                     showlegend=False))
        fig_roc.update_layout(
            height=340,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # --- Live deviation curve on a chosen run ---
    st.subheader("Per-timestep deviation: live example")
    st.markdown(
        "Pick a simulation run for this fault. The model computes per-timestep "
        "reconstruction error across the entire run. The dashed line is the "
        "anomaly threshold (99th percentile of normal-run error)."
    )

    run_id = st.number_input(
        "Run ID (1–500)", min_value=1, max_value=500, value=100, step=1,
        key=f"run_input_{fault_n}",
    )

    with st.spinner("Computing deviation curve..."):
        run_df = load_faulty_run(int(fault_n), int(run_id))
        if run_df is None:
            st.warning(f"No data for fault {fault_n} run {run_id}.")
            return

        cols = get_process_cols()
        scaler = get_scaler()
        arr = run_df[cols].to_numpy(dtype=np.float32)
        arr_scaled = scaler.transform(arr).astype(np.float32)

        model, device = load_autoencoder()
        err = _per_timestep_error(model, arr_scaled, device)

    threshold = 0.00942  # estimated in extract_anomaly_windows.py

    fig_dev = go.Figure()
    fig_dev.add_trace(go.Scatter(
        x=np.arange(len(err)), y=err, mode="lines",
        line=dict(color="#c44536", width=1.5), name="reconstruction error",
    ))
    fig_dev.add_hline(
        y=threshold, line=dict(color="#3d4f5d", dash="dash"),
        annotation_text=f"threshold = {threshold:.4f}",
        annotation_position="top left",
    )
    # Mark fault injection (sample 160 in test runs)
    fig_dev.add_vline(
        x=160, line=dict(color="#1f3a5f", dash="dot"),
        annotation_text="fault injected (sample 160)", annotation_position="top right",
    )
    fig_dev.update_layout(
        height=380,
        xaxis_title="sample (timestep)",
        yaxis_title="per-timestep reconstruction error",
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    # Stats
    above = err >= threshold
    pct_above = float(above.mean()) * 100
    first_persistent = None
    run_len = 0
    for i, v in enumerate(above):
        if v:
            run_len += 1
            if run_len >= 10:
                first_persistent = i - 9
                break
        else:
            run_len = 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Max error in run", f"{err.max():.4f}")
    s2.metric("% above threshold", f"{pct_above:.1f}%")
    s3.metric(
        "First persistent onset",
        f"sample {first_persistent}" if first_persistent is not None else "none",
    )