"""
Build faulty test arrays — memory-safe, streaming per fault type.

Strategy:
  - Subsample to N_RUNS_PER_FAULT runs per fault type (default 50)
  - Process one fault type at a time, save as X_test_faulty_fXX.npy
  - Combine metadata into a single test_faulty_meta.npz at the end

This keeps peak RAM at ~200 MB instead of 10 GB.

Saves under data/processed/:
  X_test_faulty_f01.npy ... X_test_faulty_f20.npy
  test_faulty_meta.npz   (fault_number, run_id, start_sample, file_offsets)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from windowing import build_windows, apply_scaler, load_scaler

WINDOW_SIZE = 50
STRIDE = 10
N_RUNS_PER_FAULT = 50   # subsample for tractable PCMCI later
SEED = 42

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

print(f"Window size: {WINDOW_SIZE}, stride: {STRIDE}")
print(f"Runs per fault type: {N_RUNS_PER_FAULT}\n")

scaler = load_scaler(PROC / "scaler.joblib")
print(f"Loaded scaler ({scaler.n_features_in_} features)\n")

print("Loading faulty test parquet (one-time, ~9.6M rows) ...")
faulty = pd.read_parquet(PROC / "TEP_Faulty_Testing.parquet", engine="fastparquet")
print(f"  rows: {len(faulty):,}")

fault_types = sorted(faulty["faultNumber"].unique().astype(int))
print(f"  fault types: {fault_types}\n")

rng = np.random.default_rng(SEED)

all_fault_meta: list[np.ndarray] = []
all_run_meta: list[np.ndarray] = []
all_start_meta: list[np.ndarray] = []
file_offsets: dict[int, tuple[int, int]] = {}  # fault -> (start_idx, end_idx) into combined meta
running_offset = 0

for fault_n in fault_types:
    sub = faulty[faulty["faultNumber"] == fault_n]
    runs = sorted(sub["simulationRun"].unique().astype(int))
    rng.shuffle(runs)
    chosen = set(runs[:N_RUNS_PER_FAULT])
    sub = sub[sub["simulationRun"].isin(chosen)]

    w = build_windows(sub, WINDOW_SIZE, STRIDE)
    if len(w) == 0:
        print(f"  fault {fault_n:02d}: no windows, skipping")
        continue

    X = apply_scaler(w.X, scaler)
    out_path = PROC / f"X_test_faulty_f{fault_n:02d}.npy"
    np.save(out_path, X)

    n = len(w)
    file_offsets[fault_n] = (running_offset, running_offset + n)
    running_offset += n

    all_fault_meta.append(w.fault_number)
    all_run_meta.append(w.run_id)
    all_start_meta.append(w.start_sample)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  fault {fault_n:02d}: {n:>5,} windows  ->  {out_path.name} ({size_mb:.1f} MB)")

    # explicit cleanup to release memory before next iteration
    del sub, w, X

# Combined meta file (small, fits easily in RAM)
np.savez(
    PROC / "test_faulty_meta.npz",
    fault_number=np.concatenate(all_fault_meta),
    run_id=np.concatenate(all_run_meta),
    start_sample=np.concatenate(all_start_meta),
    file_offsets=np.array(
        [(k, v[0], v[1]) for k, v in sorted(file_offsets.items())], dtype=np.int64
    ),
)

print(f"\nTotal faulty test windows across all fault types: {running_offset:,}")
print(f"Combined metadata saved: test_faulty_meta.npz")
print("\nDone.")