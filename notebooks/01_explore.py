"""
Phase 2 quick exploration:
  - Check column names and dtypes
  - Plot a single normal run vs a faulty run for one variable
  - Verify run length and sample counts match expectations
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# Load smaller files first
ff_train = pd.read_parquet(PROC / "TEP_FaultFree_Training.parquet", engine="fastparquet")
faulty_train = pd.read_parquet(PROC / "TEP_Faulty_Training.parquet", engine="fastparquet")

print("=" * 60)
print("COLUMN STRUCTURE")
print("=" * 60)
print(f"\nNumber of columns: {ff_train.shape[1]}")
print(f"\nFirst 5 column names: {list(ff_train.columns[:5])}")
print(f"Last 5 column names:  {list(ff_train.columns[-5:])}")
print(f"\nDtypes:\n{ff_train.dtypes.value_counts()}")

print("\n" + "=" * 60)
print("SAMPLE DATA — first 3 rows of normal training")
print("=" * 60)
print(ff_train.head(3).to_string())

print("\n" + "=" * 60)
print("BASIC STATS — process variables (xmeas_1 to xmeas_5)")
print("=" * 60)
xmeas_cols = [c for c in ff_train.columns if c.startswith("xmeas_")][:5]
print(ff_train[xmeas_cols].describe().round(3))

# Pick one normal run and one faulty run to plot
print("\n" + "=" * 60)
print("PLOTTING: normal run #1 vs fault-1 run #1, variable xmeas_1")
print("=" * 60)
normal = ff_train[ff_train["simulationRun"] == 1].sort_values("sample")
fault1 = faulty_train[
    (faulty_train["faultNumber"] == 1) & (faulty_train["simulationRun"] == 1)
].sort_values("sample")

print(f"  Normal run length: {len(normal)} samples")
print(f"  Fault-1 run length: {len(fault1)} samples")

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(normal["sample"], normal["xmeas_1"], color="tab:blue", lw=0.8)
axes[0].set_title("Normal operation — xmeas_1 (A feed flow)")
axes[0].set_ylabel("xmeas_1")
axes[0].grid(alpha=0.3)

axes[1].plot(fault1["sample"], fault1["xmeas_1"], color="tab:red", lw=0.8)
axes[1].axvline(20, color="black", ls="--", alpha=0.5, label="fault injection (sample 20)")
axes[1].set_title("Fault 1 (A/C feed ratio step) — xmeas_1")
axes[1].set_xlabel("sample")
axes[1].set_ylabel("xmeas_1")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).resolve().parent / "01_normal_vs_fault1.png"
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved to: {out_path}")
print("Done.")