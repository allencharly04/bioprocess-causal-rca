"""
Phase 4.1: Find deviation-onset windows in anomalous TEP runs.

For each (fault_type, simulationRun) in the faulty test set:
  1. Reconstruct the full run, get per-timestep error
  2. Find first timestep where error stays above threshold for >= MIN_PERSIST steps
  3. Extract WINDOW_LEN consecutive timesteps starting from onset for PCMCI input

The threshold is set to the 99th percentile of normal-test per-timestep error.
Output is a single .npz containing all extracted windows + their metadata,
ready to feed into PCMCI in Phase 4.2.

Outputs (under data/processed/):
  anomaly_windows.npz
    X:               (n, WINDOW_LEN, 52)  scaled deviation windows
    fault_number:    (n,)
    run_id:          (n,)
    onset_sample:    (n,)
    error_at_onset:  (n,)
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))
from model import LSTMAutoencoder
from windowing import PROCESS_COLS, apply_scaler, load_scaler

# ---- Config ----
WINDOW_LEN = 100         # length of deviation window fed to PCMCI
MIN_PERSIST = 10         # min consecutive timesteps above threshold to count as onset
N_RUNS_PER_FAULT = 5     # how many anomalous runs per fault for causal analysis
THRESHOLD_PERCENTILE = 99
SEED = 42

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
BATCH = 64


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
    return model


@torch.no_grad()
def per_timestep_error_for_run(
    model: LSTMAutoencoder,
    run_array: np.ndarray,   # (T, 52) scaled
    device: torch.device,
    window_size: int = 50,
    stride: int = 1,
) -> np.ndarray:
    """
    Slide a window across a run, compute per-timestep error, then average
    overlapping predictions to get a single (T,) error series.
    """
    T = run_array.shape[0]
    if T < window_size:
        return np.zeros(T, dtype=np.float32)

    starts = np.arange(0, T - window_size + 1, stride)
    windows = np.stack([run_array[s : s + window_size] for s in starts], axis=0)
    ds = TensorDataset(torch.from_numpy(windows).float())
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, pin_memory=(device.type == "cuda"))

    err_acc = np.zeros(T, dtype=np.float64)
    cnt_acc = np.zeros(T, dtype=np.int64)
    cursor = 0
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        ts_err = model.reconstruction_error(xb, "per_timestep").cpu().numpy()  # (B, window_size)
        for i in range(ts_err.shape[0]):
            s = starts[cursor + i]
            err_acc[s : s + window_size] += ts_err[i]
            cnt_acc[s : s + window_size] += 1
        cursor += ts_err.shape[0]

    cnt_acc[cnt_acc == 0] = 1
    return (err_acc / cnt_acc).astype(np.float32)


def find_onset(err: np.ndarray, threshold: float, min_persist: int) -> int | None:
    """First index where err >= threshold for min_persist consecutive steps. None if never."""
    above = err >= threshold
    run_len = 0
    for i, v in enumerate(above):
        if v:
            run_len += 1
            if run_len >= min_persist:
                return i - min_persist + 1
        else:
            run_len = 0
    return None


def estimate_threshold(model: LSTMAutoencoder, device: torch.device) -> float:
    """Threshold = THRESHOLD_PERCENTILE of per-timestep error on normal test runs."""
    ff_test = pd.read_parquet(PROC / "TEP_FaultFree_Testing.parquet", engine="fastparquet")
    scaler = load_scaler(PROC / "scaler.joblib")

    # Sample 50 normal runs to estimate threshold (full 500 is overkill)
    rng = np.random.default_rng(SEED)
    runs = sorted(ff_test["simulationRun"].unique().astype(int))
    rng.shuffle(runs)
    chosen = runs[:50]

    all_errs: list[np.ndarray] = []
    print(f"  estimating threshold on {len(chosen)} normal runs ...")
    for run_id in chosen:
        run_df = ff_test[ff_test["simulationRun"] == run_id].sort_values("sample")
        arr = run_df[PROCESS_COLS].to_numpy(dtype=np.float32)
        arr = apply_scaler(arr[None, :, :], scaler)[0]   # scale via shape (1,T,52)
        err = per_timestep_error_for_run(model, arr, device)
        all_errs.append(err)
    all_errs_flat = np.concatenate(all_errs)
    thr = float(np.percentile(all_errs_flat, THRESHOLD_PERCENTILE))
    print(f"  normal per-timestep error: mean={all_errs_flat.mean():.5f}, "
          f"p{THRESHOLD_PERCENTILE}={thr:.5f}")
    return thr


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")
    print(f"[setup] window_len={WINDOW_LEN}, min_persist={MIN_PERSIST}, "
          f"runs/fault={N_RUNS_PER_FAULT}\n")

    print("[load] model ...")
    model = load_model(device)

    print("\n[threshold] estimating from normal runs ...")
    threshold = estimate_threshold(model, device)
    print(f"  -> threshold = {threshold:.5f}")

    print("\n[load] faulty test parquet ...")
    faulty = pd.read_parquet(PROC / "TEP_Faulty_Testing.parquet", engine="fastparquet")
    scaler = load_scaler(PROC / "scaler.joblib")
    fault_types = sorted(faulty["faultNumber"].unique().astype(int))

    rng = np.random.default_rng(SEED)
    extracted_X: list[np.ndarray] = []
    extracted_fnum: list[int] = []
    extracted_run: list[int] = []
    extracted_onset: list[int] = []
    extracted_err_at_onset: list[float] = []
    skipped: list[tuple[int, int, str]] = []

    print(f"\n[extract] looking for deviation onsets ...")
    for fnum in fault_types:
        sub_runs = sorted(
            faulty.loc[faulty["faultNumber"] == fnum, "simulationRun"].unique().astype(int)
        )
        rng.shuffle(sub_runs)

        accepted_for_fault = 0
        for run_id in sub_runs:
            if accepted_for_fault >= N_RUNS_PER_FAULT:
                break
            run_df = faulty[
                (faulty["faultNumber"] == fnum) & (faulty["simulationRun"] == run_id)
            ].sort_values("sample")
            arr = run_df[PROCESS_COLS].to_numpy(dtype=np.float32)
            arr = apply_scaler(arr[None, :, :], scaler)[0]   # (T, 52)

            err = per_timestep_error_for_run(model, arr, device)
            onset = find_onset(err, threshold, MIN_PERSIST)
            if onset is None:
                skipped.append((fnum, run_id, "no onset"))
                continue
            if onset + WINDOW_LEN > arr.shape[0]:
                skipped.append((fnum, run_id, f"onset@{onset} too late"))
                continue

            window = arr[onset : onset + WINDOW_LEN]   # (WINDOW_LEN, 52)
            extracted_X.append(window)
            extracted_fnum.append(fnum)
            extracted_run.append(run_id)
            extracted_onset.append(onset)
            extracted_err_at_onset.append(float(err[onset]))
            accepted_for_fault += 1

        print(f"  fault {fnum:02d}: accepted {accepted_for_fault}/{N_RUNS_PER_FAULT}")

    if not extracted_X:
        print("\n[error] no anomaly windows extracted!")
        return

    X = np.stack(extracted_X, axis=0)
    out_path = PROC / "anomaly_windows.npz"
    np.savez(
        out_path,
        X=X,
        fault_number=np.array(extracted_fnum, dtype=np.int64),
        run_id=np.array(extracted_run, dtype=np.int64),
        onset_sample=np.array(extracted_onset, dtype=np.int64),
        error_at_onset=np.array(extracted_err_at_onset, dtype=np.float32),
        threshold=np.float32(threshold),
        process_cols=np.array(PROCESS_COLS),
    )
    print(f"\n[save] {out_path.name}")
    print(f"  total anomaly windows: {len(X)}")
    print(f"  shape: {X.shape}")
    print(f"  fault counts: "
          f"{ {f: int(np.sum(np.array(extracted_fnum) == f)) for f in fault_types} }")
    if skipped:
        print(f"  skipped runs: {len(skipped)} (first few: {skipped[:5]})")


if __name__ == "__main__":
    main()