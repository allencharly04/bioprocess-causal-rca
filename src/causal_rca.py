"""
Phase 4.2: PCMCI causal discovery on anomaly windows.

For each anomaly window:
  1. Run PCMCI with ParCorr test, tau_max=5
  2. Build the time-lagged causal graph (significant links at p<0.05)
  3. Rank variables by (out-degree - in-degree) — root-cause candidates
  4. Compare top-K to documented TEP ground truth

Outputs (under models/):
  causal_results.npz       per-window: ranking, top-3 vars, hit/miss
  causal_summary.csv       per-fault: top-1 / top-3 hit rate
  causal_graphs/fXX_runYY.png   visualization of strongest causal links per window
"""

from __future__ import annotations
from pathlib import Path
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

sys.path.append(str(Path(__file__).resolve().parent))
from windowing import PROCESS_COLS

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
GRAPHS_DIR = MODELS / "causal_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

TAU_MAX = 5
PC_ALPHA = 0.05
ALPHA_LEVEL = 0.05


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


def rank_root_causes(p_matrix: np.ndarray, val_matrix: np.ndarray, alpha: float):
    """Rank variables by out-degree minus in-degree in the lagged causal graph."""
    N = p_matrix.shape[0]
    sig = (p_matrix < alpha) & np.isfinite(val_matrix)

    out_deg = np.zeros(N, dtype=np.float64)
    in_deg = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for tau in range(1, sig.shape[2]):
                if sig[i, j, tau]:
                    out_deg[i] += 1
                    in_deg[j] += 1

    score = out_deg - in_deg
    return np.argsort(-score), score, out_deg, in_deg


def run_pcmci_on_window(window: np.ndarray):
    df = pp.DataFrame(window, var_names=PROCESS_COLS)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA_LEVEL)
    return results["p_matrix"], results["val_matrix"]


def plot_causal_graph(val_matrix, p_matrix, top_k_vars, title, out_path, alpha=ALPHA_LEVEL):
    sig = (p_matrix < alpha) & np.isfinite(val_matrix)
    agg = np.zeros((len(top_k_vars), len(top_k_vars)), dtype=np.float64)
    for ai, i in enumerate(top_k_vars):
        for bi, j in enumerate(top_k_vars):
            if i == j:
                continue
            mask = sig[i, j, 1:]
            if mask.any():
                agg[ai, bi] = np.max(np.abs(val_matrix[i, j, 1:][mask]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [PROCESS_COLS[v] for v in top_k_vars]
    im = ax.imshow(agg, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("driven variable (effect)")
    ax.set_ylabel("driving variable (cause)")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="max |partial correlation|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    print(f"[setup] tau_max={TAU_MAX}, pc_alpha={PC_ALPHA}, alpha_level={ALPHA_LEVEL}\n")

    print("[load] anomaly windows ...")
    data = np.load(PROC / "anomaly_windows.npz", allow_pickle=True)
    X = data["X"]
    fault_number = data["fault_number"]
    run_id = data["run_id"]
    onset = data["onset_sample"]
    print(f"  {len(X)} windows, shape {X.shape}\n")

    n = len(X)
    rankings = np.zeros((n, len(PROCESS_COLS)), dtype=np.int64)
    top1_hit = np.zeros(n, dtype=bool)
    top3_hit = np.zeros(n, dtype=bool)
    scores_all = np.zeros((n, len(PROCESS_COLS)), dtype=np.float64)

    for i in range(n):
        fnum = int(fault_number[i])
        rid = int(run_id[i])
        gt_set = GROUND_TRUTH.get(fnum, set())
        try:
            p_mat, v_mat = run_pcmci_on_window(X[i])
        except Exception as e:
            print(f"  [{i+1:>3}/{n}] fault {fnum:02d} run {rid}: PCMCI failed ({e})")
            continue

        ranked, score, out_deg, in_deg = rank_root_causes(p_mat, v_mat, ALPHA_LEVEL)
        rankings[i] = ranked
        scores_all[i] = score

        top1 = int(ranked[0])
        top3 = ranked[:3].tolist()
        hit1 = top1 in gt_set
        hit3 = any(v in gt_set for v in top3)
        top1_hit[i] = hit1
        top3_hit[i] = hit3

        top1_name = PROCESS_COLS[top1]
        gt_names = [PROCESS_COLS[v] for v in gt_set] if gt_set else ["?"]
        tag1 = "HIT" if hit1 else "   "
        tag3 = "HIT" if hit3 else "   "
        print(f"  [{i+1:>3}/{n}] f{fnum:02d} r{rid:>3}: "
              f"top1={top1_name:<8} top3={[PROCESS_COLS[v] for v in top3]}  "
              f"gt={gt_names}  [{tag1}/{tag3}]")

        if i == 0 or fault_number[i] != fault_number[i-1]:
            top_vars = list(ranked[:8])
            for v in gt_set:
                if v not in top_vars:
                    top_vars.append(v)
            plot_path = GRAPHS_DIR / f"f{fnum:02d}_run{rid:03d}.png"
            plot_causal_graph(
                v_mat, p_mat, top_vars,
                f"Fault {fnum} (run {rid}) — causal links among top-ranked variables",
                plot_path,
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
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
    print(f"\n  OVERALL: top1={overall_t1:.3f}  top3={overall_t3:.3f}  (n={n} windows)")

    csv_path = MODELS / "causal_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("fault_number,n_windows,top1_acc,top3_acc\n")
        for fnum, n_f, t1, t3 in rows:
            f.write(f"{fnum},{n_f},{t1:.4f},{t3:.4f}\n")
        f.write(f"OVERALL,{n},{overall_t1:.4f},{overall_t3:.4f}\n")
    print(f"\n[save] {csv_path.name}")

    np.savez(
        MODELS / "causal_results.npz",
        rankings=rankings,
        scores=scores_all,
        top1_hit=top1_hit,
        top3_hit=top3_hit,
        fault_number=fault_number,
        run_id=run_id,
        onset_sample=onset,
        process_cols=np.array(PROCESS_COLS),
    )
    print("[save] causal_results.npz")
    print(f"[save] {GRAPHS_DIR}/  ({len(list(GRAPHS_DIR.glob('*.png')))} causal graph PNGs)")
    print("\n[done]")


if __name__ == "__main__":
    main()