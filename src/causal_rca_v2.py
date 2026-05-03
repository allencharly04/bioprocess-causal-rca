"""
Phase 4.2 v2: PCMCI causal discovery + improved ranking.

Improvements over v1:
  1. Pre-filter candidate variables to top-K by autoencoder per-feature
     reconstruction error on the deviation window (only deviating variables
     can be root causes).
  2. Score by sum of |partial correlation| on significant lag-1 outgoing
     edges (not just edge count).
  3. Cache full PCMCI output (p_matrix, val_matrix) per window so future
     ranker iterations don't require re-running PCMCI.

Outputs (under models/):
  causal_results_v2.npz             rankings, hits, plus p_mats/val_mats CACHED
  causal_summary_v2.csv             per-fault top-1 / top-3 hit rate
  causal_graphs_v2/fXX_runYY.png    causal heatmaps among ranked top vars
"""

from __future__ import annotations
from pathlib import Path
import sys
import warnings
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

sys.path.append(str(Path(__file__).resolve().parent))
from windowing import PROCESS_COLS
from model import LSTMAutoencoder

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
GRAPHS_DIR = MODELS / "causal_graphs_v2"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

TAU_MAX = 5
PC_ALPHA = 0.05
ALPHA_LEVEL = 0.05
TOP_K_DEVIANT = 15        # filter to top-15 deviating variables before ranking


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
def per_feature_error(model: LSTMAutoencoder, window: np.ndarray, device: torch.device) -> np.ndarray:
    """
    window: (T, F) where T >= 50.
    Slide 50-step windows, get per-feature error per window, average -> (F,).
    """
    T, F = window.shape
    starts = np.arange(0, T - 50 + 1, 1)
    sub = np.stack([window[s:s+50] for s in starts], axis=0).astype(np.float32)
    x = torch.from_numpy(sub).to(device)
    err = model.reconstruction_error(x, "per_feature").cpu().numpy()  # (n_subwin, F)
    return err.mean(axis=0)


def rank_with_filter(
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    candidate_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score each candidate variable by sum of |val| for significant
    OUTGOING LAG-1 edges to any other candidate.
    Returns (ranked_indices_in_full_52_space, score_per_candidate_in_candidate_order).
    """
    sig = (p_matrix < alpha) & np.isfinite(val_matrix)
    cand = list(candidate_idx)
    scores = np.zeros(len(cand), dtype=np.float64)
    for ai, i in enumerate(cand):
        for j in cand:
            if i == j:
                continue
            # lag-1 only
            if sig[i, j, 1]:
                scores[ai] += abs(val_matrix[i, j, 1])

    order_in_cand = np.argsort(-scores)
    ranked_global = np.array([cand[k] for k in order_in_cand], dtype=np.int64)
    return ranked_global, scores


def run_pcmci(window: np.ndarray):
    df = pp.DataFrame(window, var_names=PROCESS_COLS)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA_LEVEL)
    return results["p_matrix"], results["val_matrix"]


def plot_top_graph(val_matrix, p_matrix, top_vars, title, out_path, alpha=ALPHA_LEVEL):
    sig = (p_matrix < alpha) & np.isfinite(val_matrix)
    agg = np.zeros((len(top_vars), len(top_vars)), dtype=np.float64)
    for ai, i in enumerate(top_vars):
        for bi, j in enumerate(top_vars):
            if i == j:
                continue
            if sig[i, j, 1]:
                agg[ai, bi] = abs(val_matrix[i, j, 1])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [PROCESS_COLS[v] for v in top_vars]
    im = ax.imshow(agg, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("driven (effect)"); ax.set_ylabel("driving (cause)")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="|partial corr|, lag-1, p<0.05")
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close(fig)


def main() -> None:
    print(f"[setup] tau_max={TAU_MAX}, top_k_deviant={TOP_K_DEVIANT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}\n")

    print("[load] anomaly windows + autoencoder ...")
    data = np.load(PROC / "anomaly_windows.npz", allow_pickle=True)
    X = data["X"]
    fault_number = data["fault_number"]
    run_id = data["run_id"]
    onset = data["onset_sample"]
    print(f"  {len(X)} windows, shape {X.shape}")

    ae = load_autoencoder(device)

    n = len(X)
    F = len(PROCESS_COLS)
    rankings = np.zeros((n, F), dtype=np.int64)
    candidate_sets = np.zeros((n, TOP_K_DEVIANT), dtype=np.int64)
    top1_hit = np.zeros(n, dtype=bool)
    top3_hit = np.zeros(n, dtype=bool)
    p_mats = np.zeros((n, F, F, TAU_MAX + 1), dtype=np.float32)
    v_mats = np.zeros((n, F, F, TAU_MAX + 1), dtype=np.float32)

    t_start = time.time()
    for i in range(n):
        fnum = int(fault_number[i])
        rid = int(run_id[i])
        gt_set = GROUND_TRUTH.get(fnum, set())

        # Per-feature deviation -> top-K candidate filter
        feat_err = per_feature_error(ae, X[i], device)
        candidates = np.argsort(-feat_err)[:TOP_K_DEVIANT]
        candidate_sets[i] = candidates

        try:
            p_mat, v_mat = run_pcmci(X[i])
        except Exception as e:
            print(f"  [{i+1:>3}/{n}] f{fnum:02d} r{rid}: PCMCI failed ({e})")
            continue

        p_mats[i] = p_mat.astype(np.float32)
        v_mats[i] = v_mat.astype(np.float32)

        ranked, scores = rank_with_filter(p_mat, v_mat, candidates, ALPHA_LEVEL)
        # Pad ranked to length F (rest with non-candidates appended in feat-err order)
        non_cand = [k for k in np.argsort(-feat_err) if k not in set(ranked.tolist())]
        full_ranking = np.concatenate([ranked, np.array(non_cand, dtype=np.int64)])[:F]
        rankings[i] = full_ranking

        top1 = int(full_ranking[0])
        top3 = full_ranking[:3].tolist()
        hit1 = top1 in gt_set
        hit3 = any(v in gt_set for v in top3)
        top1_hit[i] = hit1
        top3_hit[i] = hit3

        elapsed = time.time() - t_start
        eta_min = (elapsed / (i + 1)) * (n - i - 1) / 60
        gt_names = [PROCESS_COLS[v] for v in gt_set] if gt_set else ["?"]
        print(f"  [{i+1:>3}/{n}] f{fnum:02d} r{rid:>3}: "
              f"top1={PROCESS_COLS[top1]:<9} "
              f"top3={[PROCESS_COLS[v] for v in top3]}  "
              f"gt={gt_names}  "
              f"[{'HIT' if hit1 else '   '}/{'HIT' if hit3 else '   '}]  "
              f"ETA {eta_min:.1f}min")

        if i == 0 or fault_number[i] != fault_number[i - 1]:
            top_vars = list(full_ranking[:8])
            for v in gt_set:
                if v not in top_vars:
                    top_vars.append(v)
            plot_top_graph(
                v_mat, p_mat, top_vars,
                f"Fault {fnum} (run {rid}) — lag-1 causal links among ranked vars",
                GRAPHS_DIR / f"f{fnum:02d}_run{rid:03d}.png",
            )

    print("\n" + "=" * 60)
    print("SUMMARY (v2: deviation filter + edge-strength scoring + lag-1)")
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

    csv_path = MODELS / "causal_summary_v2.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("fault_number,n_windows,top1_acc,top3_acc\n")
        for fnum, n_f, t1, t3 in rows:
            f.write(f"{fnum},{n_f},{t1:.4f},{t3:.4f}\n")
        f.write(f"OVERALL,{n},{overall_t1:.4f},{overall_t3:.4f}\n")
    print(f"\n[save] {csv_path.name}")

    # Cache PCMCI matrices so we can re-rank in seconds without rerunning
    np.savez_compressed(
        MODELS / "causal_results_v2.npz",
        rankings=rankings,
        candidate_sets=candidate_sets,
        top1_hit=top1_hit,
        top3_hit=top3_hit,
        fault_number=fault_number,
        run_id=run_id,
        onset_sample=onset,
        p_mats=p_mats,
        v_mats=v_mats,
        process_cols=np.array(PROCESS_COLS),
    )
    print("[save] causal_results_v2.npz (with cached p_mats/v_mats)")
    print(f"[save] {GRAPHS_DIR}/  ({len(list(GRAPHS_DIR.glob('*.png')))} PNGs)")
    print("\n[done]")


if __name__ == "__main__":
    main()