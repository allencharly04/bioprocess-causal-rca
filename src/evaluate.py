"""
Evaluate the trained autoencoder on TEP test sets.

Computes reconstruction error per window for:
  - Normal test windows (negative class)
  - Faulty test windows, per fault type 1..20 (positive classes)

Outputs (under models/):
  eval_scores.npz       per-window scores + labels
  eval_auc.csv          per-fault ROC-AUC table
  eval_score_dist.png   score distributions (normal vs each fault)
  eval_roc_curves.png   ROC curves
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from model import LSTMAutoencoder

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
BATCH = 256


def load_model(device: torch.device) -> LSTMAutoencoder:
    ckpt = torch.load(MODELS / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LSTMAutoencoder(
        n_features=52,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        window_size=50,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded weights from epoch {ckpt['epoch']}, val_mse={ckpt['val_mse']:.6f}")
    return model


@torch.no_grad()
def compute_scores(model: LSTMAutoencoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return per-window reconstruction error, shape (N,)."""
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, pin_memory=(device.type == "cuda"))
    scores: list[np.ndarray] = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        err = model.reconstruction_error(xb, "per_window")
        scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")

    print("[load] model ...")
    model = load_model(device)

    # --- Normal test ---
    print("\n[score] normal test windows ...")
    X_normal = np.load(PROC / "X_test_normal.npy")
    scores_normal = compute_scores(model, X_normal, device)
    print(f"  {len(scores_normal):,} windows, mean error: {scores_normal.mean():.4f}, "
          f"p99: {np.percentile(scores_normal, 99):.4f}")

    # --- Faulty test, per fault ---
    print("\n[score] faulty test windows, per fault type ...")
    fault_files = sorted(PROC.glob("X_test_faulty_f*.npy"))
    per_fault: dict[int, np.ndarray] = {}
    for fp in fault_files:
        fnum = int(fp.stem.split("_f")[-1])
        X_f = np.load(fp)
        s = compute_scores(model, X_f, device)
        per_fault[fnum] = s
        print(f"  fault {fnum:02d}: {len(s):,} windows, mean error: {s.mean():.4f}")

    # --- ROC-AUC per fault: normal=0, faulty=1 ---
    print("\n[metric] per-fault ROC-AUC ...")
    auc_rows: list[tuple[int, int, float]] = []
    for fnum in sorted(per_fault.keys()):
        s_f = per_fault[fnum]
        y_true = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(s_f))])
        y_score = np.concatenate([scores_normal, s_f])
        auc = roc_auc_score(y_true, y_score)
        auc_rows.append((fnum, len(s_f), auc))
        print(f"  fault {fnum:02d}: AUC = {auc:.4f}")

    aucs = [r[2] for r in auc_rows]
    print(f"\n  mean AUC across faults: {np.mean(aucs):.4f}")
    print(f"  median AUC:             {np.median(aucs):.4f}")

    # --- Save AUC CSV ---
    csv_path = MODELS / "eval_auc.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("fault_number,n_windows,roc_auc\n")
        for fnum, n, auc in auc_rows:
            f.write(f"{fnum},{n},{auc:.4f}\n")
    print(f"\n[save] {csv_path.name}")

    # --- Save raw scores ---
    np.savez(
        MODELS / "eval_scores.npz",
        normal=scores_normal,
        **{f"fault_{k:02d}": v for k, v in per_fault.items()},
    )
    print(f"[save] eval_scores.npz")

    # --- Score distribution plot ---
    fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=False, sharey=False)
    bins = np.linspace(0, max(scores_normal.max(), max(s.max() for s in per_fault.values())), 60)
    for ax, fnum in zip(axes.flat, sorted(per_fault.keys())):
        ax.hist(scores_normal, bins=bins, alpha=0.5, label="normal", color="tab:blue", density=True)
        ax.hist(per_fault[fnum], bins=bins, alpha=0.5, label=f"fault {fnum}", color="tab:red", density=True)
        auc = dict((r[0], r[2]) for r in auc_rows)[fnum]
        ax.set_title(f"Fault {fnum:02d}  AUC={auc:.3f}", fontsize=10)
        ax.set_xlim(0, np.percentile(np.concatenate([scores_normal, per_fault[fnum]]), 99))
        ax.tick_params(labelsize=8)
    fig.suptitle("Reconstruction error: normal (blue) vs faulty (red), per fault type", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dist_path = MODELS / "eval_score_dist.png"
    plt.savefig(dist_path, dpi=110)
    plt.close(fig)
    print(f"[save] {dist_path.name}")

    # --- ROC curves plot ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for fnum in sorted(per_fault.keys()):
        s_f = per_fault[fnum]
        y_true = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(s_f))])
        y_score = np.concatenate([scores_normal, s_f])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = dict((r[0], r[2]) for r in auc_rows)[fnum]
        ax.plot(fpr, tpr, lw=1.0, alpha=0.8, label=f"f{fnum:02d} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — LSTM autoencoder anomaly detection")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = MODELS / "eval_roc_curves.png"
    plt.savefig(roc_path, dpi=120)
    plt.close(fig)
    print(f"[save] {roc_path.name}")

    print("\n[done]")


if __name__ == "__main__":
    main()