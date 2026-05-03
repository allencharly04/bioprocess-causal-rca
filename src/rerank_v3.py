"""
Phase 4.3: v3 reranker — hub-aware scoring on cached PCMCI matrices.

Loads cached p_mats/v_mats from causal_results_v2.npz, then re-ranks using:
  1. Lags 1..3 outgoing edge strengths (not just lag-1)
  2. Hub penalty: penalize variables that show up in many windows' top-K
  3. Deviation magnitude weighting from autoencoder per-feature error

Pure re-ranking — no PCMCI rerun. Runs in under a minute.

Outputs:
  models/causal_summary_v3.csv
  models/causal_results_v3.npz
  models/causal_graphs_v3/fXX_runYY.png
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from windowing import PROCESS_COLS
from model import LSTMAutoencoder

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
GRAPHS_DIR = MODELS / "causal_graphs_v3"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
LAGS_USED = [1, 2, 3]       # which lags contribute to outgoing-edge score
ALPHA_LEVEL = 0.05
HUB_PENALTY_WEIGHT = 0.6    # 0 = no penalty, 1 = full penalty
TOP_K_FOR_HUB_DETECTION = 5 # define "hub" as variable in top-5 of many windows
EPS = 1e-9


def name_to_idx(name: str) -> int:
    return PROCESS_COLS.index(name)

GROUND_TRUTH: dict[int, set[int]] = {
    1:  {name_to_idx("xmeas_1"),  name_to_idx("xmeas_4"),  name_to_idx("xmv_3")},
    2:  {name_to_idx("xmeas_4")},
    3:  {name_to_idx("xmeas_2")},
    4:  {name_to_idx("xmeas_9")},
    5:  {name_to_idx("xmeas_22")},
    6:  {name_to_idx("xmeas_1"),  name_to_idx("xmv_3")},
    7:  {name_to_idx("xmeas_4"),  name_to_idx("xmv_4")},
    8:  {name_to_idx("xmeas_4")},
    9:  {name_to_idx("xmeas_2")},
    10: {name_to_idx("xmeas_18")},
    11: {name_to_idx("xmeas_9")},
    12: {name_to_idx("xmeas_22")},
    13: {name_to_idx("xmeas_9")},
    14: {name_to_idx("xmv_10")},
    15: {name_to_idx("xmv_11")},
    16: {name_to_idx("xmeas_9")},
    17: {name_to_idx("xmeas_9")},
    18: {name_to_idx("xmeas_9")},
    19: {name_to_idx("xmv_10")},
    20: {name_to_idx("xmeas_18")},
}


def load_autoencoder(device: torch.device) -> LSTMAutoencoder:
    ckpt = torch.load(MODELS / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LSTMAutoencoder(
        n_features=52, hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"], window_size=50, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def per_feature_error_for_window(model: LSTMAutoencoder, window: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-feature mean reconstruction error for one (T, F) window."""
    T, F = window.shape
    starts = np.arange(0, T - 50 + 1, 1)
    sub = np.stack([window[s:s+50] for s in starts], axis=0).astype(np.float32)
    x = torch.from_numpy(sub).to(device)
    err = model.reconstruction_error(x, "per_feature").cpu().numpy()  # (n_subwin, F)
    return err.mean(axis=0)


def causal_strength_per_var(p_mat: np.ndarray, v_mat: np.ndarray, lags=LAGS_USED, alpha=ALPHA_LEVEL) -> np.ndarray:
    """
    For each variable i, sum |val[i, j, tau]| over j != i and tau in lags
    where the link is significant. Returns shape (N,).
    """
    sig = (p_mat < alpha) & np.isfinite(v_mat)
    N = p_mat.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    for tau in lags:
        # outgoing for variable i = sum over j of |v_mat[i, j, tau]| where significant
        mask_tau = sig[:, :, tau]                       # (N, N)
        np.fill_diagonal(mask_tau, False)
        contrib = np.where(mask_tau, np.abs(v_mat[:, :, tau]), 0.0)
        scores += contrib.sum(axis=1)                    # row sum per source variable
    return scores


def compute_hub_frequency(p_mats: np.ndarray, v_mats: np.ndarray) -> np.ndarray:
    """
    Pre-pass: for each window, compute v2-style ranking on raw causal strength.
    Then count how often each variable appears in the top-K_FOR_HUB_DETECTION
    across all windows. Return a normalized [0, 1] hub frequency vector.
    """
    n_windows = p_mats.shape[0]
    N = p_mats.shape[1]
    counts = np.zeros(N, dtype=np.float64)

    for i in range(n_windows):
        s = causal_strength_per_var(p_mats[i], v_mats[i])
        top_k = np.argsort(-s)[:TOP_K_FOR_HUB_DETECTION]
        counts[top_k] += 1

    return counts / max(n_windows, 1)


def rank_v3(
    p_mat: np.ndarray, v_mat: np.ndarray,
    feat_err: np.ndarray, hub_freq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    v3 score = (causal outgoing strength) * (deviation magnitude) - hub penalty.

    All three components are normalized to [0, 1] within this window/global so
    HUB_PENALTY_WEIGHT is interpretable.
    """
    causal = causal_strength_per_var(p_mat, v_mat)

    # Normalize each component to [0, 1]
    def normalize(x: np.ndarray) -> np.ndarray:
        m, M = x.min(), x.max()
        return (x - m) / (M - m + EPS)

    causal_n = normalize(causal)
    feat_n = normalize(feat_err)
    hub_n = hub_freq  # already in [0, 1] by construction

    score = causal_n * feat_n - HUB_PENALTY_WEIGHT * hub_n
    ranked = np.argsort(-score)
    return ranked, score


def plot_top_graph(v_mat, p_mat, top_vars, title, out_path, alpha=ALPHA_LEVEL, lags=LAGS_USED):
    sig = (p_mat < alpha) & np.isfinite(v_mat)
    agg = np.zeros((len(top_vars), len(top_vars)), dtype=np.float64)
    for ai, i in enumerate(top_vars):
        for bi, j in enumerate(top_vars):
            if i == j:
                continue
            best = 0.0
            for tau in lags:
                if sig[i, j, tau]:
                    best = max(best, abs(v_mat[i, j, tau]))
            agg[ai, bi] = best

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [PROCESS_COLS[v] for v in top_vars]
    im = ax.imshow(agg, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("driven (effect)"); ax.set_ylabel("driving (cause)")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=f"max |partial corr|, lags 1-3, p<{alpha}")
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close(fig)


def main() -> None:
    print(f"[setup] lags={LAGS_USED}, hub_penalty={HUB_PENALTY_WEIGHT}, "
          f"top_k_hub_detection={TOP_K_FOR_HUB_DETECTION}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")

    print("[load] cached PCMCI results + autoencoder + anomaly windows ...")
    cached = np.load(MODELS / "causal_results_v2.npz", allow_pickle=True)
    p_mats = cached["p_mats"]
    v_mats = cached["v_mats"]
    fault_number = cached["fault_number"]
    run_id = cached["run_id"]
    onset = cached["onset_sample"]

    aw = np.load(PROC / "anomaly_windows.npz", allow_pickle=True)
    X = aw["X"]

    n = len(p_mats)
    F = len(PROCESS_COLS)
    print(f"  {n} windows, {F} features\n")

    ae = load_autoencoder(device)

    print("[step 1] computing per-window per-feature deviation magnitude ...")
    feat_errs = np.zeros((n, F), dtype=np.float64)
    for i in range(n):
        feat_errs[i] = per_feature_error_for_window(ae, X[i], device)
    print(f"  done.\n")

    print(f"[step 2] computing global hub frequency from {n} cached windows ...")
    hub_freq = compute_hub_frequency(p_mats, v_mats)
    top_hubs = np.argsort(-hub_freq)[:8]
    print(f"  top hub variables (frequency in top-{TOP_K_FOR_HUB_DETECTION}):")
    for h in top_hubs:
        print(f"    {PROCESS_COLS[h]:<10}: {hub_freq[h]:.2%}")
    print()

    print("[step 3] re-ranking each window with v3 scoring ...")
    rankings = np.zeros((n, F), dtype=np.int64)
    scores_all = np.zeros((n, F), dtype=np.float64)
    top1_hit = np.zeros(n, dtype=bool)
    top3_hit = np.zeros(n, dtype=bool)

    for i in range(n):
        fnum = int(fault_number[i])
        rid = int(run_id[i])
        gt_set = GROUND_TRUTH.get(fnum, set())

        ranked, score = rank_v3(p_mats[i], v_mats[i], feat_errs[i], hub_freq)
        rankings[i] = ranked
        scores_all[i] = score

        top1 = int(ranked[0])
        top3 = ranked[:3].tolist()
        top1_hit[i] = top1 in gt_set
        top3_hit[i] = any(v in gt_set for v in top3)

        gt_names = [PROCESS_COLS[v] for v in gt_set] if gt_set else ["?"]
        tag1 = "HIT" if top1_hit[i] else "   "
        tag3 = "HIT" if top3_hit[i] else "   "
        print(f"  [{i+1:>3}/{n}] f{fnum:02d} r{rid:>3}: "
              f"top1={PROCESS_COLS[top1]:<9} "
              f"top3={[PROCESS_COLS[v] for v in top3]}  "
              f"gt={gt_names}  [{tag1}/{tag3}]")

        if i == 0 or fault_number[i] != fault_number[i - 1]:
            top_vars = list(ranked[:8])
            for v in gt_set:
                if v not in top_vars:
                    top_vars.append(v)
            plot_top_graph(
                v_mats[i], p_mats[i], top_vars,
                f"Fault {fnum} (run {rid}) - v3 lags 1-3 causal links",
                GRAPHS_DIR / f"f{fnum:02d}_run{rid:03d}.png",
            )

    print("\n" + "=" * 60)
    print("SUMMARY (v3: lags 1-3 + hub penalty + deviation weighting)")
    print("=" * 60)

    rows = []
    for fnum in sorted(set(int(f) for f in fault_number)):
        mask = (fault_number == fnum)
        n_f = int(mask.sum())
        t1 = float(top1_hit[mask].mean()) if n_f else 0.0
        t3 = float(top3_hit[mask].mean()) if n_f else 0.0
        rows.append((fnum, n_f, t1, t3))
        print(f"  fault {fnum:02d}: n={n_f}  top1={t1:.2f}  top3={t3:.2f}")

    overall_t1 = float(top1_hit.mean())
    overall_t3 = float(top3_hit.mean())
    print(f"\n  OVERALL: top1={overall_t1:.3f}  top3={overall_t3:.3f}  (n={n})")
    print(f"\n  vs v2: top1=0.130 top3=0.250")
    print(f"  vs v1: top1=0.109 top3=0.141")

    csv_path = MODELS / "causal_summary_v3.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("fault_number,n_windows,top1_acc,top3_acc\n")
        for fnum, n_f, t1, t3 in rows:
            f.write(f"{fnum},{n_f},{t1:.4f},{t3:.4f}\n")
        f.write(f"OVERALL,{n},{overall_t1:.4f},{overall_t3:.4f}\n")
    print(f"\n[save] {csv_path.name}")

    np.savez(
        MODELS / "causal_results_v3.npz",
        rankings=rankings,
        scores=scores_all,
        top1_hit=top1_hit,
        top3_hit=top3_hit,
        fault_number=fault_number,
        run_id=run_id,
        onset_sample=onset,
        hub_freq=hub_freq,
        feat_errs=feat_errs,
        process_cols=np.array(PROCESS_COLS),
    )
    print("[save] causal_results_v3.npz")
    print(f"[save] {GRAPHS_DIR}/  ({len(list(GRAPHS_DIR.glob('*.png')))} PNGs)")
    print("\n[done]")


if __name__ == "__main__":
    main()