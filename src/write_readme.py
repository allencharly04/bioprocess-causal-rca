"""
One-shot script to write a clean README.md to the project root.
Bypasses notepad's encoding issues with multi-line markdown.
"""

from pathlib import Path

README = r"""# ProcessRCA

**Causal Root Cause Analysis for Multivariate Industrial Time-Series**

A digital-twin extension that combines deep-learning anomaly detection with time-series causal discovery to identify *which* sensor drove a process deviation — not just *that* a deviation occurred. Validated on the Tennessee Eastman Process benchmark (52 sensors, 20 fault types, 15.3M timesteps).

![Dashboard overview](models/dashboard_hero.png)

---

## TL;DR

Standard ML on industrial sensor data tells you **when** something is wrong. It rarely tells you **which sensor caused it** — operators detect a deviation in seconds, but root-cause analysis can take hours. ProcessRCA closes that gap with a four-stage pipeline: an LSTM autoencoder detects deviations, a threshold rule localizes onset, PCMCI infers a time-lagged causal graph, and a hub-aware ranker proposes the top-K root causes.

| Metric | Result |
|---|---|
| Median detection ROC-AUC (LSTM autoencoder) | **0.91** across 20 faults |
| Strong-detection faults (AUC ≥ 0.85) | **17 / 20** |
| Causal RCA top-3 hit rate over 92 windows | **0.29** (random baseline = 0.058) |
| Faults reaching ≥ 80% top-3 hit rate | **8 / 20** (1, 10, 11, 12, 14, 17 + others) |

---

## Pipeline

```mermaid
flowchart TD
    A[Sensor data<br/>52 vars × time] --> B
    B[<b>Stage 1 · Detection</b><br/>LSTM autoencoder<br/>trained on normal only<br/><i>→ reconstruction error</i>] --> C
    C[<b>Stage 2 · Localization</b><br/>Threshold + persistence rule<br/>10 consecutive timesteps<br/><i>→ deviation onset</i>] --> D
    D[<b>Stage 3 · Causal discovery</b><br/>PCMCI + ParCorr<br/>τ_max = 5<br/><i>→ time-lagged causal graph</i>] --> E
    E[<b>Stage 4 · Ranking</b><br/>Hub-aware scoring<br/>causal × deviation − hub_penalty<br/><i>→ top-K root causes</i>]

    style B fill:#1f3a5f,color:#fff,stroke:#1f3a5f
    style C fill:#2b5d6e,color:#fff,stroke:#2b5d6e
    style D fill:#2b7a78,color:#fff,stroke:#2b7a78
    style E fill:#c44536,color:#fff,stroke:#c44536
```

---

## Why this exists

In a real industrial plant — a chemical reactor, a pharma fermenter, a power station — when something goes wrong, dozens of sensors deviate at once. The variables are *all* highly correlated within minutes. A correlation-based method (feature importance, attention, autoencoder error alone) will flag many of them, but cannot distinguish a true cause from a downstream effect that just happens to be highly correlated. Only a method that separates **causal direction** can identify the actual source.

ProcessRCA does this on the **Tennessee Eastman Process** — the canonical 30-year-old chemical-engineering benchmark used in hundreds of fault-detection papers — and validates against documented ground-truth root causes from the original Downs & Vogel (1993) plant model.

---

## Headline result: ranker iteration

The single most informative artifact in this project. The ranker went through three principled iterations, each addressing a specific failure mode of the previous version:

![Ranker comparison](models/ranker_comparison_v1v2v3.png)

| Version | Key change | Top-3 |
|---|---|---|
| v1 | Score = (out-edges − in-edges), all lags, no filter | 0.141 |
| v2 | + autoencoder-deviation pre-filter to top-15 candidates; score by ∑\|partial corr\| on lag-1 | 0.250 |
| v3 | + multi-lag (1–3) edges and **hub penalty** based on global frequency in top-K | **0.293** |
| Random baseline (3 out of 52 vars) | — | 0.058 |

**v3 catches the actual root cause in the top-3 about 5× more often than chance.** On 8 of 20 fault types (40%), it hits top-3 in ≥ 80% of windows.

---

## Detection: per-fault ROC-AUC

| Tier | Faults | AUC range |
|---|---|---|
| Strong | 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 19, 20 | 0.89 – 0.93 |
| Moderate | 18 | 0.89 |
| Weak (incipient) | 3, 9, 15 | 0.55 – 0.62 |

Faults 3, 9, and 15 are the well-documented "incipient" faults — subtle drifts that are famously difficult to detect. Russell, Chiang & Braatz (2000) and every subsequent paper report similar weakness here. The pipeline correctly produces fewer detectable windows for these faults, which is honest behavior — RCA can only operate where detection succeeds.

See per-fault diagnostic plots in [`models/per_fault_diagnostics/`](models/per_fault_diagnostics/).

---

## Interactive dashboard

A multi-page Streamlit app (`app/`) lets a user explore the project end-to-end:

- **Overview** — pipeline diagram, headline metrics, per-fault top-3 chart, training curve
- **Fault Detection** — pick a fault type, see ROC + score distribution + a live deviation curve on any of 500 simulation runs
- **Causal RCA** — pick an anomaly window, see predicted top-5 root causes, ground-truth comparison, per-feature deviation chart, top-5 variable time-series, and the global hub-frequency table

Run with:
```bash
streamlit run app/app.py
```

---

## Stack

| Layer | Tools |
|---|---|
| Data | pandas, fastparquet, pyreadr (.RData) |
| Modeling | PyTorch 2.5.1 + CUDA 12.1, AMP mixed precision |
| Causal | tigramite 5.2 (PCMCI + ParCorr), networkx, dowhy |
| Visualization | matplotlib, plotly |
| UI | Streamlit |

---

## Reproducing the results

```bash
# 1. Environment
conda create -n tep-causal python=3.11 -y
conda activate tep-causal
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Data — download 4 .RData files from
#    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
#    and place them in data/raw/

# 3. Pipeline
python src/load_tep.py                       # parse .RData -> parquet
python src/build_arrays.py                   # train/val/normal-test arrays
python src/build_faulty_arrays.py            # per-fault test arrays
python src/train.py                          # ~2 min on RTX 2060
python src/evaluate.py                       # per-fault ROC-AUC
python src/extract_anomaly_windows.py        # deviation onset detection
python src/causal_rca_v2.py                  # ~2 hr - PCMCI on 92 windows
python src/rerank_v3.py                      # ~30 sec - re-rank from cache

# 4. Visualization
python src/plot_v1v2v3.py
python src/plot_per_fault_diagnostics.py

# 5. Dashboard
streamlit run app/app.py
```

Total compute time end-to-end on an RTX 2060: ~2.5 hours, almost all in the PCMCI step. PCMCI matrices are cached after first run, so iterating on the ranker is sub-second.

---

## Project structure
bioprocess-causal-rca/
├── README.md
├── data/
│   ├── raw/                      .RData files (download from Harvard Dataverse)
│   └── processed/                parquet, .npy, scaler, anomaly_windows.npz
├── src/
│   ├── windowing.py              windowing utilities + scaler helpers
│   ├── load_tep.py               .RData → parquet
│   ├── build_arrays.py           train/val/normal-test arrays
│   ├── build_faulty_arrays.py    per-fault test arrays (memory-safe)
│   ├── model.py                  LSTM autoencoder
│   ├── train.py                  training loop + curves
│   ├── evaluate.py               per-fault ROC-AUC + score distributions
│   ├── extract_anomaly_windows.py
│   ├── causal_rca.py             ranker v1 (baseline)
│   ├── causal_rca_v2.py          ranker v2 + cache PCMCI matrices
│   ├── rerank_v3.py              ranker v3 (hub-aware) on cached matrices
│   ├── plot_v1v2v3.py
│   └── plot_per_fault_diagnostics.py
├── models/
│   ├── best_model.pt             trained autoencoder weights
│   ├── training_curve.png
│   ├── eval_auc.csv
│   ├── ranker_comparison_v1v2v3.png
│   ├── causal_summary_v1.csv, _v2.csv, _v3.csv
│   ├── causal_results_v3.npz     rankings + scores (PCMCI cache excluded)
│   └── per_fault_diagnostics/    20 PNGs, one per fault type
├── app/
│   ├── app.py                    Streamlit entry point
│   ├── pages_overview.py
│   ├── pages_detection.py
│   ├── pages_rca.py
│   └── data_loader.py            cached loaders
└── notebooks/
└── 01_explore.py             initial data exploration

---

## Honest limitations

This project does **not** claim:

- **State-of-the-art on TEP.** Specialized published methods report higher numbers. The goal here is a clean, reproducible pipeline that demonstrates the method, not a leaderboard entry.
- **Generalization without retraining.** The autoencoder and threshold are calibrated to TEP. Applying to a new process requires re-training on its normal-operation data.
- **Real-time operation.** PCMCI takes ~60 seconds per window; suitable for post-incident analysis, not millisecond control loops.
- **Identification of unknown faults.** Validation requires ground-truth root causes. The method ranks candidates; a domain expert still verifies.

The hub-aware v3 ranker also mildly regressed on Fault 12 (1.00 → 0.40 top-3) because its true root cause (xmeas_22) happens to be a frequent hub variable — exactly the kind of variable the penalty downweights. This is a genuine trade-off and is documented as future work.

---

## What I learned building this

1. **Architecture matters more than hyperparameters.** Switching from a bottleneck-vector LSTM autoencoder to a sequence-to-sequence variant dropped val MSE 230× — worth more than any amount of learning-rate tuning.
2. **Memory blows up faster than you expect.** Holding 920k windows of (50, 52) float32 in RAM during scaling caused a system freeze. Streaming per-fault and aggressive cleanup cut peak RAM by 10×.
3. **Cache expensive computations early.** PCMCI takes 2 hours; saving the raw output matrices means subsequent ranking iterations cost seconds, not hours.
4. **Honest evaluation beats inflated numbers.** Reporting AUC 0.55 on incipient faults — with citations to prior work showing the same weakness — is more credible than glossing over it.
5. **Causal ≠ correlational.** Watching the v1 ranker (count-based) latch onto hub variables made the difference between a method that almost works and one that does work viscerally clear.

---

## References

- Downs, J. J. & Vogel, E. F. (1993). *A plant-wide industrial process control problem.* Computers & Chemical Engineering, 17(3), 245–255.
- Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse, V1. https://doi.org/10.7910/DVN/6C3JR1
- Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). *Detecting and quantifying causal associations in large nonlinear time series datasets.* Science Advances, 5(11).
- Russell, E. L., Chiang, L. H., & Braatz, R. D. (2000). *Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis.* Chemometrics and Intelligent Laboratory Systems, 51(1), 81–93.
"""

out = Path(__file__).resolve().parent.parent / "README.md"
out.write_text(README, encoding="utf-8")
print(f"Wrote: {out}")
print(f"  size: {out.stat().st_size / 1024:.1f} KB")