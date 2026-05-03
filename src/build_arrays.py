"""
Build train/val/normal-test arrays. Memory-safe.

Skips faulty test (handled by build_faulty_arrays.py).
Reuses an existing scaler.joblib if found, else fits one.

Saves under data/processed/:
  X_train.npy, X_val.npy, X_test_normal.npy
  test_normal_meta.npz
  scaler.joblib
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from windowing import (
    build_windows, fit_scaler, apply_scaler, save_scaler, load_scaler,
    PROCESS_COLS,
)

WINDOW_SIZE = 50
STRIDE = 10
VAL_FRAC = 0.2
SEED = 42

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

print(f"Window size: {WINDOW_SIZE}, stride: {STRIDE}\n")

# ---- 1. Normal training: split per-run, window, scale, save ----
print("[1/3] Normal training -> train + val ...")
ff_train = pd.read_parquet(PROC / "TEP_FaultFree_Training.parquet", engine="fastparquet")

rng = np.random.default_rng(SEED)
all_runs = sorted(ff_train["simulationRun"].unique().astype(int))
rng.shuffle(all_runs)
n_val = int(len(all_runs) * VAL_FRAC)
val_runs = set(all_runs[:n_val])
train_runs = set(all_runs[n_val:])

train_df = ff_train[ff_train["simulationRun"].isin(train_runs)]
val_df = ff_train[ff_train["simulationRun"].isin(val_runs)]
del ff_train

train_w = build_windows(train_df, WINDOW_SIZE, STRIDE)
val_w = build_windows(val_df, WINDOW_SIZE, STRIDE)
del train_df, val_df
print(f"  train: {train_w.X.shape}, val: {val_w.X.shape}")

# Fit or reuse scaler
scaler_path = PROC / "scaler.joblib"
if scaler_path.exists():
    print(f"  reusing existing scaler: {scaler_path.name}")
    scaler = load_scaler(scaler_path)
else:
    print("  fitting new scaler on train windows ...")
    scaler = fit_scaler(train_w.X)
    save_scaler(scaler, scaler_path)

X_train = apply_scaler(train_w.X, scaler)
np.save(PROC / "X_train.npy", X_train)
del train_w, X_train
print(f"  saved X_train.npy")

X_val = apply_scaler(val_w.X, scaler)
np.save(PROC / "X_val.npy", X_val)
del val_w, X_val
print(f"  saved X_val.npy")

# ---- 2. Normal test ----
print("\n[2/3] Normal test ...")
ff_test = pd.read_parquet(PROC / "TEP_FaultFree_Testing.parquet", engine="fastparquet")
test_normal_w = build_windows(ff_test, WINDOW_SIZE, STRIDE)
del ff_test
print(f"  shape: {test_normal_w.X.shape}")

X_test_normal = apply_scaler(test_normal_w.X, scaler)
np.save(PROC / "X_test_normal.npy", X_test_normal)
np.savez(
    PROC / "test_normal_meta.npz",
    fault_number=test_normal_w.fault_number,
    run_id=test_normal_w.run_id,
    start_sample=test_normal_w.start_sample,
)
del test_normal_w, X_test_normal
print("  saved X_test_normal.npy + test_normal_meta.npz")

# ---- 3. Summary ----
print("\n[3/3] Summary of files in data/processed/:")
for f in sorted(PROC.glob("X_*.npy")):
    print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
print("\nDone. Run build_faulty_arrays.py next for the faulty test set.")