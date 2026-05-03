"""
Generate a per-fault comparison plot of top-3 hit rate across ranker versions.

The v1 numbers are hardcoded from the original v1 run output.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

V1_TOP3 = {
    1: 0.20, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.20,
    6: 0.00, 7: 0.40, 8: 0.00, 9: 0.00, 10: 0.20,
    11: 0.00, 12: 1.00, 13: 0.00, 14: 0.00, 15: 0.00,
    16: 0.00, 17: 0.00, 18: 0.20, 19: 0.20, 20: 0.20,
}


def load(version: str) -> dict[int, float]:
    df = pd.read_csv(MODELS / f"causal_summary_{version}.csv")
    df = df[df["fault_number"] != "OVERALL"].copy()
    df["fault_number"] = df["fault_number"].astype(int)
    df["top3_acc"] = df["top3_acc"].astype(float)
    return dict(zip(df["fault_number"], df["top3_acc"]))


def main() -> None:
    v2 = load("v2")
    v3 = load("v3")
    faults = sorted(set(V1_TOP3) | set(v2) | set(v3))

    v1_vals = [V1_TOP3.get(f, 0.0) for f in faults]
    v2_vals = [v2.get(f, 0.0) for f in faults]
    v3_vals = [v3.get(f, 0.0) for f in faults]

    x = np.arange(len(faults))
    w = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - w, v1_vals, w, label="v1: count-based", color="#a85a5a")
    ax.bar(x,       v2_vals, w, label="v2: + deviation filter", color="#c4a35a")
    ax.bar(x + w, v3_vals, w, label="v3: + hub penalty, lags 1-3", color="#2b7a78")

    ax.set_xticks(x)
    ax.set_xticklabels([f"f{f:02d}" for f in faults], rotation=0, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Fault type", fontsize=11)
    ax.set_ylabel("Top-3 hit rate", fontsize=11)
    ax.set_title(
        "Causal RCA top-3 hit rate by fault type - ranker iteration v1 -> v2 -> v3",
        fontsize=12, pad=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    overall_text = (
        "Overall top-3 hit rate:\n"
        "  v1 = 0.141\n"
        "  v2 = 0.250  (+77%)\n"
        "  v3 = 0.293  (+108% vs v1)"
    )
    ax.text(
        0.985, 0.97, overall_text,
        transform=ax.transAxes,
        fontsize=9.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8", ec="#d6dde3"),
        family="monospace",
    )

    plt.tight_layout()
    out = MODELS / "ranker_comparison_v1v2v3.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()