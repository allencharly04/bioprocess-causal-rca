"""
TEP windowing utilities.

Builds (window_size, n_features) tensors from per-run time-series.
Critical: windows NEVER cross run boundaries (each simulationRun is independent).

The 52 process variables are: xmeas_1..xmeas_41 + xmv_1..xmv_11.
The 3 metadata columns (faultNumber, simulationRun, sample) are excluded
from the model input but kept for downstream filtering and labeling.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 52 process variables (the model's actual inputs)
PROCESS_COLS: list[str] = (
    [f"xmeas_{i}" for i in range(1, 42)] +
    [f"xmv_{i}" for i in range(1, 12)]
)
META_COLS = ["faultNumber", "simulationRun", "sample"]


@dataclass
class WindowedData:
    """Container for a windowed dataset."""
    X: np.ndarray              # (n_windows, window_size, n_features), float32
    fault_number: np.ndarray   # (n_windows,) int — fault label of source run
    run_id: np.ndarray         # (n_windows,) int — simulationRun index
    start_sample: np.ndarray   # (n_windows,) int — first sample index of window

    def __len__(self) -> int:
        return len(self.X)


def make_windows_for_run(
    run_df: pd.DataFrame,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a single sorted-by-sample run.

    Returns:
        windows: (n, window_size, n_features) float32
        starts:  (n,) int — first 'sample' index of each window
    """
    arr = run_df[PROCESS_COLS].to_numpy(dtype=np.float32)
    samples = run_df["sample"].to_numpy()
    n_steps = len(arr)
    if n_steps < window_size:
        return np.empty((0, window_size, len(PROCESS_COLS)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts_idx = np.arange(0, n_steps - window_size + 1, stride)
    windows = np.stack([arr[i : i + window_size] for i in starts_idx], axis=0)
    starts = samples[starts_idx]
    return windows, starts


def build_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 10,
) -> WindowedData:
    """Build windows from a TEP DataFrame, respecting run boundaries."""
    all_X: list[np.ndarray] = []
    all_fault: list[np.ndarray] = []
    all_run: list[np.ndarray] = []
    all_start: list[np.ndarray] = []

    # group by (faultNumber, simulationRun) — each is one independent run
    grouped = df.sort_values(["faultNumber", "simulationRun", "sample"]).groupby(
        ["faultNumber", "simulationRun"], sort=False
    )
    for (fault_n, run_n), run_df in grouped:
        windows, starts = make_windows_for_run(run_df, window_size, stride)
        if len(windows) == 0:
            continue
        all_X.append(windows)
        all_fault.append(np.full(len(windows), int(fault_n), dtype=np.int64))
        all_run.append(np.full(len(windows), int(run_n), dtype=np.int64))
        all_start.append(starts.astype(np.int64))

    return WindowedData(
        X=np.concatenate(all_X, axis=0),
        fault_number=np.concatenate(all_fault),
        run_id=np.concatenate(all_run),
        start_sample=np.concatenate(all_start),
    )


def fit_scaler(windows: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on flattened windows (each feature scaled independently)."""
    n, w, f = windows.shape
    flat = windows.reshape(-1, f)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, w, f = windows.shape
    flat = windows.reshape(-1, f)
    scaled = scaler.transform(flat).astype(np.float32)
    return scaled.reshape(n, w, f)


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    import joblib
    return joblib.load(path)