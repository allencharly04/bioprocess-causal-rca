"""Causal RCA page — explore individual anomaly windows + ranked root causes."""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_loader import (
    load_causal_results, load_anomaly_windows, get_process_cols,
    FAULT_DESCRIPTIONS,
)


# Ground truth: same mapping as causal_rca_v2/v3
def _build_ground_truth(cols: list[str]) -> dict[int, set[int]]:
    def idx(name: str) -> int: return cols.index(name)
    return {
        1:  {idx("xmeas_1"),  idx("xmeas_4"),  idx("xmv_3")},
        2:  {idx("xmeas_4")},
        3:  {idx("xmeas_2")},
        4:  {idx("xmeas_9")},
        5:  {idx("xmeas_22")},
        6:  {idx("xmeas_1"),  idx("xmv_3")},
        7:  {idx("xmeas_4"),  idx("xmv_4")},
        8:  {idx("xmeas_4")},
        9:  {idx("xmeas_2")},
        10: {idx("xmeas_18")},
        11: {idx("xmeas_9")},
        12: {idx("xmeas_22")},
        13: {idx("xmeas_9")},
        14: {idx("xmv_10")},
        15: {idx("xmv_11")},
        16: {idx("xmeas_9")},
        17: {idx("xmeas_9")},
        18: {idx("xmeas_9")},
        19: {idx("xmv_10")},
        20: {idx("xmeas_18")},
    }


def render() -> None:
    st.title("Causal Root Cause Analysis")
    st.markdown(
        "Each anomaly window is the deviation segment of one faulty simulation run. "
        "PCMCI infers a time-lagged causal graph over all 52 process variables. "
        "The ranker combines the autoencoder's per-feature deviation with PCMCI's "
        "outgoing-edge strength, penalizing variables that act as 'hubs' across many faults. "
        "Use the controls to inspect any window."
    )

    cols = get_process_cols()
    GT = _build_ground_truth(cols)

    res = load_causal_results("v3")
    aw = load_anomaly_windows()

    rankings = res["rankings"]              # (n, 52)
    fault_number = res["fault_number"]      # (n,)
    run_id = res["run_id"]                  # (n,)
    onset = res["onset_sample"]             # (n,)
    feat_errs = res["feat_errs"]            # (n, 52)
    hub_freq = res["hub_freq"]              # (52,)
    top1_hit = res["top1_hit"]              # (n,)
    top3_hit = res["top3_hit"]              # (n,)
    X_windows = aw["X"]                     # (n, T, 52)

    n = len(rankings)

    # --- Sidebar-style controls in main panel ---
    available_faults = sorted(set(int(f) for f in fault_number))
    col1, col2 = st.columns([1, 1])
    with col1:
        fault_pick = st.selectbox(
            "Fault type", available_faults,
            format_func=lambda f: f"Fault {int(f):02d}",
        )
    with col2:
        # Find indices for this fault, then pick a run
        idx_for_fault = np.where(fault_number == fault_pick)[0]
        run_options = [int(run_id[i]) for i in idx_for_fault]
        run_pick = st.selectbox("Run", run_options, format_func=lambda r: f"Run {int(r)}")

    selected_idx = int(np.where(
        (fault_number == fault_pick) & (run_id == run_pick)
    )[0][0])

    # --- Header / context ---
    st.markdown("---")
    st.markdown(
        f"**Fault {int(fault_pick):02d}**: "
        f"{FAULT_DESCRIPTIONS.get(int(fault_pick), '?')}  &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Run**: {int(run_pick)}  &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Deviation onset**: sample {int(onset[selected_idx])}"
    )

    gt_set = GT.get(int(fault_pick), set())
    gt_names = sorted(cols[v] for v in gt_set)

    # Predicted ranks
    ranking = rankings[selected_idx]
    top5 = [int(v) for v in ranking[:5]]
    top5_names = [cols[v] for v in top5]

    # --- Top-5 prediction card + ground truth comparison ---
    pred_col, gt_col = st.columns(2)

    with pred_col:
        st.subheader("Predicted top-5 root causes")
        for rank, (var_idx, name) in enumerate(zip(top5, top5_names), start=1):
            is_match = var_idx in gt_set
            badge = "✓ ground truth" if is_match else ""
            colour = "#2b7a78" if is_match else "#3d4f5d"
            st.markdown(
                f"<div style='padding:6px 10px; margin-bottom:6px; "
                f"background:#f4f6f8; border-left:3px solid {colour}; border-radius:3px;'>"
                f"<b>#{rank}</b> &nbsp; <code>{name}</code> "
                f"<span style='float:right; color:{colour};'>{badge}</span>"
                f"</div>", unsafe_allow_html=True,
            )

    with gt_col:
        st.subheader("Ground truth")
        if gt_names:
            st.markdown(
                "Documented root-cause variable(s) for this fault, from the "
                "Downs & Vogel TEP specification:"
            )
            for n_gt in gt_names:
                in_top1 = (top5[0] == cols.index(n_gt))
                in_top3 = (cols.index(n_gt) in top5[:3])
                if in_top1:
                    tag = "in TOP-1"
                    colour = "#2b7a78"
                elif in_top3:
                    tag = "in TOP-3"
                    colour = "#2b7a78"
                else:
                    tag = "not in top-3"
                    colour = "#a85a5a"
                st.markdown(
                    f"<div style='padding:6px 10px; margin-bottom:6px; "
                    f"background:#f4f6f8; border-left:3px solid {colour}; border-radius:3px;'>"
                    f"<code>{n_gt}</code> "
                    f"<span style='float:right; color:{colour};'>{tag}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("(no documented ground truth)")

        hit1 = bool(top1_hit[selected_idx])
        hit3 = bool(top3_hit[selected_idx])
        st.markdown(
            f"**This window**: "
            f"top-1 {'HIT' if hit1 else 'miss'}, "
            f"top-3 {'HIT' if hit3 else 'miss'}"
        )

    st.markdown("---")

    # --- Per-feature deviation bar chart (top 15) ---
    st.subheader("Per-feature deviation (top 15)")
    st.markdown(
        "Reconstruction error per process variable in this deviation window. "
        "High bars = variables that the model cannot reconstruct well, i.e. "
        "variables actively deviating from normal behavior."
    )
    fe = feat_errs[selected_idx]
    order = np.argsort(-fe)[:15]
    bar_names = [cols[i] for i in order]
    bar_vals = fe[order]
    bar_colors = []
    for i in order:
        if i == top5[0]:
            bar_colors.append("#1f3a5f")        # top-1 prediction
        elif i in top5[:3]:
            bar_colors.append("#2b7a78")        # in top-3 prediction
        elif i in gt_set:
            bar_colors.append("#c44536")        # ground truth not predicted
        else:
            bar_colors.append("#9aa5b1")

    fig_fe = go.Figure(go.Bar(
        x=bar_names, y=bar_vals, marker_color=bar_colors,
        text=[f"{v:.3f}" for v in bar_vals], textposition="outside",
    ))
    fig_fe.update_layout(
        height=360,
        yaxis_title="reconstruction error",
        margin=dict(l=20, r=20, t=20, b=80),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_fe, use_container_width=True)
    st.caption(
        "Navy = predicted top-1 · teal = predicted top-3 · "
        "red = ground-truth (not in our top-3) · grey = other deviating variables"
    )

    st.markdown("---")

    # --- Time-series view of top variables in this window ---
    st.subheader("Time-series of top deviating variables")
    st.markdown(
        "Scaled time-series for the top-5 most-deviating variables in this 100-step "
        "deviation window."
    )
    window = X_windows[selected_idx]   # (T, 52), already scaled
    T = window.shape[0]

    fig_ts = go.Figure()
    palette = ["#1f3a5f", "#2b7a78", "#c44536", "#c4a35a", "#7a5b9c"]
    for color, var_idx in zip(palette, order[:5]):
        fig_ts.add_trace(go.Scatter(
            x=np.arange(T), y=window[:, var_idx], mode="lines",
            name=cols[var_idx], line=dict(width=1.6, color=color),
        ))
    fig_ts.update_layout(
        height=360,
        xaxis_title="timestep within deviation window",
        yaxis_title="scaled value (z-score)",
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # --- Hub frequencies (interpretation aid) ---
    st.subheader("Why hub-aware ranking matters")
    st.markdown(
        "The ranker penalizes variables that appear as drivers across many faults — "
        "they are likely 'hubs' in the plant's natural feedback loops, not specific "
        "root causes. The top hub variables (across all 92 anomaly windows) are:"
    )
    hub_order = np.argsort(-hub_freq)[:8]
    hub_df = pd.DataFrame({
        "variable": [cols[i] for i in hub_order],
        "appears in top-5 across all faults": [f"{hub_freq[i]:.0%}" for i in hub_order],
    })
    st.dataframe(hub_df, use_container_width=True, hide_index=True)