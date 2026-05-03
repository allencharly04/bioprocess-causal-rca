"""
Load TEP .RData files, inspect structure, and save as parquet for faster reloads.

Each .RData file contains an R dataframe. Schema (per the Rieth 2017 docs):
  Column 1: faultNumber     (0 = normal, 1-20 = fault type)
  Column 2: simulationRun   (1-500, RNG seed index)
  Column 3: sample          (1-500 for training, 1-960 for testing)
  Columns 4-44: xmeas_1 to xmeas_41   (41 process measurements)
  Columns 45-55: xmv_1 to xmv_11      (11 manipulated variables)

Total: 52 process variables + 3 metadata columns = 55 columns.
"""

from pathlib import Path
import pyreadr
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

FILES = {
    "TEP_FaultFree_Training.RData": "fault_free_training",
    "TEP_FaultFree_Testing.RData":  "fault_free_testing",
    "TEP_Faulty_Training.RData":    "faulty_training",
    "TEP_Faulty_Testing.RData":     "faulty_testing",
}


def load_and_save(filename: str, r_var_name: str) -> pd.DataFrame:
    src = RAW / filename
    dst = PROC / (filename.replace(".RData", ".parquet"))

    if dst.exists():
        print(f"[cached] {dst.name} already exists, loading parquet ...")
        df = pd.read_parquet(dst, engine="fastparquet")
    else:
        print(f"[parse] {filename} ({src.stat().st_size / 1e6:.1f} MB) ...")
        result = pyreadr.read_r(str(src))
        # pyreadr returns an OrderedDict of {name: DataFrame}
        if r_var_name in result:
            df = result[r_var_name]
        else:
            # fall back: take the only dataframe in the file
            df = next(iter(result.values()))
        print(f"  -> {len(df):,} rows, {df.shape[1]} cols")
        print(f"[save] {dst.name} ...")
        df.to_parquet(dst, engine="fastparquet", compression="snappy")
    return df


def main() -> None:
    print(f"Raw dir: {RAW}")
    print(f"Processed dir: {PROC}\n")

    summaries = []
    for fname, rname in FILES.items():
        df = load_and_save(fname, rname)
        summaries.append({
            "file": fname,
            "rows": len(df),
            "cols": df.shape[1],
            "fault_types": sorted(df["faultNumber"].unique().tolist()),
            "n_runs": df["simulationRun"].nunique(),
            "samples_per_run": df.groupby(["faultNumber", "simulationRun"]).size().iloc[0],
        })
        print()

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for s in summaries:
        print(f"\n{s['file']}")
        print(f"  rows         : {s['rows']:,}")
        print(f"  columns      : {s['cols']}")
        print(f"  fault types  : {s['fault_types']}")
        print(f"  unique runs  : {s['n_runs']}")
        print(f"  samples/run  : {s['samples_per_run']}")


if __name__ == "__main__":
    main()