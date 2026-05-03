"""
Generate per-fault diagnostic plots for the README and PDF.

For each fault 1..20:
  - Score distribution (normal vs faulty windows)
  - ROC curve with AUC annotation
  - One example deviation curve (live autoencoder per-timestep error)
  - Top-10 most-deviating variables in a representative anomaly window

Output: models/per_fault_diagnostics/fXX.png  (one per fault)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from windowing import PROCESS_COLS, apply_scaler, load_scaler
from model import LSTMAutoencoder

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT_DIR = MODELS / "per_fault_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.00942

FAULT_DESCRIPTIONS = {
    1:  "A/C feed ratio step",
    2:  "B composition step",
    3:  "D feed temperature step",
    4:  "Reactor cooling water inlet temp step",
    5:  "Condenser cooling water inlet temp step",
    6:  "A feed loss",
    7:  "C header pressure loss",
    8:  "A,B,C feed composition random",
    9:  "D feed temperature random",
    10: "C feed temperature random",
    11: "Reactor cooling water inlet temp random",
    12: "Condenser cooling water inlet temp random",
    13: "Reaction kinetics drift",
    14: "Reactor cooling water valve sticking",
    15: "Condenser cooling water valve sticking",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
}


def load_model(device):
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
def per_timestep_error(model, run_array_scaled, device):
    T = run_array_scaled.shape[0]
    if T < 50:
        return np.zeros(T, dtype=np.float32)
    starts = np.arange(0, T - 50 + 1, 1)
    sub = np.stack([run_array_scaled[s:s+50] for s in starts], axis=0).astype(np.float32)
    err_acc = np.zeros(T, dtype=np.float64)
    cnt_acc = np.zeros(T, dtype=np.int64)
    bs = 64
    for i in range(0, len(sub), bs):
        xb = torch.from_numpy(sub[i:i+bs]).to(device)
        ts_err = model.reconstruction_error(xb, "per_timestep").cpu().numpy()
        for k in range(ts_err.shape[0]):
            s = starts[i+k]
            err_acc[s:s+50] += ts_err[k]
            cnt_acc[s:s+50] += 1
    cnt_acc[cnt_acc == 0] = 1
    return (err_acc / cnt_acc).astype(np.float32)


def make_plot(fault_n: int, eval_scores: dict, faulty_test_df, scaler, model, device):
    s_normal = eval_scores["normal"]
    s_fault = eval_scores[f"fault_{fault_n:02d}"]
    auc_val = roc_auc_score(
        np.concatenate([np.zeros(len(s_normal)), np.ones(len(s_fault))]),
        np.concatenate([s_normal, s_fault]),
    )

    # Pick a run with a clear deviation: highest mean error in first 5 runs
    runs = sorted(
        faulty_test_df[faulty_test_df["faultNumber"] == fault_n]["simulationRun"]
        .unique().astype(int)
    )[:20]
    best_run, best_err_mean, best_err_curve, best_arr = None, -1.0, None, None
    for run_id in runs:
        sub = faulty_test_df[
            (faulty_test_df["faultNumber"] == fault_n) &
            (faulty_test_df["simulationRun"] == run_id)
        ].sort_values("sample")
        if len(sub) < 100:
            continue
        arr = sub[PROCESS_COLS].to_numpy(dtype=np.float32)
        arr_scaled = scaler.transform(arr).astype(np.float32)
        err = per_timestep_error(model, arr_scaled, device)
        m = float(err[160:].mean())  # mean post-injection
        if m > best_err_mean:
            best_run, best_err_mean = int(run_id), m
            best_err_curve = err
            best_arr = arr_scaled

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. Score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cap = float(np.percentile(np.concatenate([s_normal, s_fault]), 99.5))
    bins = np.linspace(0, cap, 50)
    ax1.hist(np.clip(s_normal, 0, cap), bins=bins, alpha=0.55, color="#2b7a78",
             label="normal", density=True)
    ax1.hist(np.clip(s_fault, 0, cap), bins=bins, alpha=0.55, color="#c44536",
             label=f"fault {fault_n}", density=True)
    ax1.set_xlabel("reconstruction error per window")
    ax1.set_ylabel("density")
    ax1.set_title("Score distribution: normal vs faulty windows", fontsize=11)
    ax1.legend()
    ax1.grid(alpha=0.25)

    # 2. ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    y_true = np.concatenate([np.zeros(len(s_normal)), np.ones(len(s_fault))])
    y_score = np.concatenate([s_normal, s_fault])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax2.plot(fpr, tpr, color="#1f3a5f", lw=2, label=f"AUC = {auc_val:.3f}")
    ax2.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC curve", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    # 3. Live deviation curve
    ax3 = fig.add_subplot(gs[1, 0])
    if best_err_curve is not None:
        ax3.plot(best_err_curve, color="#c44536", lw=1.2, label=f"run {best_run}")
        ax3.axhline(THRESHOLD, color="#3d4f5d", ls="--", lw=1,
                    label=f"threshold = {THRESHOLD:.4f}")
        ax3.axvline(160, color="#1f3a5f", ls=":", lw=1, label="fault @ sample 160")
        ax3.set_xlabel("sample (timestep)")
        ax3.set_ylabel("per-timestep reconstruction error")
        ax3.set_title(f"Live deviation curve (example: run {best_run})", fontsize=11)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "no usable run", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_axis_off()

    # 4. Top-10 deviating variables in 100-step post-onset window
    ax4 = fig.add_subplot(gs[1, 1])
    if best_arr is not None and best_err_curve is not None:
        # find onset
        above = best_err_curve >= THRESHOLD
        run_len = 0; onset = None
        for i, v in enumerate(above):
            if v:
                run_len += 1
                if run_len >= 10:
                    onset = i - 9; break
            else:
                run_len = 0
        if onset is not None and onset + 100 <= best_arr.shape[0]:
            window = best_arr[onset:onset+100]
            with torch.no_grad():
                starts = np.arange(0, 100 - 50 + 1, 1)
                sub = np.stack([window[s:s+50] for s in starts], axis=0).astype(np.float32)
                xb = torch.from_numpy(sub).to(device)
                fe = model.reconstruction_error(xb, "per_feature").cpu().numpy().mean(axis=0)
            order = np.argsort(-fe)[:10]
            names = [PROCESS_COLS[i] for i in order]
            vals = fe[order]
            ax4.barh(range(10), vals[::-1], color="#2b7a78")
            ax4.set_yticks(range(10))
            ax4.set_yticklabels(names[::-1], fontsize=9)
            ax4.set_xlabel("per-feature reconstruction error")
            ax4.set_title("Top-10 deviating variables in deviation window", fontsize=11)
            ax4.grid(axis="x", alpha=0.25)
        else:
            ax4.text(0.5, 0.5, "no clear onset", ha="center", va="center",
                     transform=ax4.transAxes)
            ax4.set_axis_off()
    else:
        ax4.set_axis_off()

    fig.suptitle(
        f"Fault {fault_n:02d} — {FAULT_DESCRIPTIONS.get(fault_n, '?')}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    out = OUT_DIR / f"f{fault_n:02d}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")

    print("[load] eval scores ...")
    eval_scores = dict(np.load(MODELS / "eval_scores.npz"))

    print("[load] faulty test parquet (one-time) ...")
    faulty = pd.read_parquet(PROC / "TEP_Faulty_Testing.parquet", engine="fastparquet")

    scaler = load_scaler(PROC / "scaler.joblib")
    model = load_model(device)

    for fnum in range(1, 21):
        print(f"  fault {fnum:02d} ...", end="", flush=True)
        out = make_plot(fnum, eval_scores, faulty, scaler, model, device)
        print(f" -> {out.name} ({out.stat().st_size/1024:.0f} KB)")

    print(f"\n[done] {len(list(OUT_DIR.glob('*.png')))} PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()