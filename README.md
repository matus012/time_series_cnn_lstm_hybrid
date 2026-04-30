# NN Time Series Comparison

Comparison of **CNN**, **LSTM**, and **CNN-LSTM hybrid** models for multivariate time-series forecasting on the **Jena Climate dataset** using **PyTorch**.

## Project Goal

The goal of this project is to compare three neural network architectures for temperature prediction from multivariate weather time series:

- **1D-CNN**
- **LSTM**
- **CNN-LSTM hybrid**

We focus on:
- prediction accuracy (**RMSE**, **MAE**),
- effect of input window size (**24 vs 48**),
- number of parameters,
- training time,
- convergence stability,
- and qualitative error analysis.

## Team

- **Person 1**: Data pipeline, shared training infrastructure, LSTM
- **Person 2**: CNN
- **Person 3**: CNN-LSTM Hybrid
- **Person 4**: Evaluation, plots, analysis, report integration

## Dataset

We use the **Jena Climate Dataset** (Max Planck Institute for Biogeochemistry).

- Source: https://www.kaggle.com/datasets/mnassrib/jena-climate
- 14 features, target = `T (degC)`
- Original cadence: 10 min — we **subsample to hourly** (every 6th row)
- Chronological split **70 / 15 / 15** (train / val / test) — no shuffling
- `StandardScaler` is **fit on the training split only**, then applied to val/test
- Target stats (used for inverse transform during evaluation):
  - mean = **9.108 °C**, std = **8.655 °C**

## Repo Structure

```
project/code/
├── src/
│   ├── dataset.py            # Loading, subsampling, splits, scaling, windowing
│   ├── train.py              # Shared training loop + early stopping
│   ├── evaluate.py           # Metric computation + plots
│   ├── utils.py              # Seeds, device
│   ├── models/
│   │   ├── cnn.py
│   │   ├── lstm.py
│   │   └── hybrid.py
│   ├── run_lstm.py           # Trains LSTM configA / configB
│   ├── run_cnn.py            # Trains CNN configA / configB
│   ├── run_hybrid.py         # Trains Hybrid configA / configB
│   └── run_evaluation.py     # Loads checkpoints, evaluates, builds plots
├── results/                  # Checkpoints, JSON metrics, plots (generated)
├── archive/                  # Frozen run logs and master results table
├── requirements.txt
└── README.md
```

The `report/` directory is hand-in material — do not modify from code.

## Setup

Python **3.11** on Windows. From `project/code/`:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The dataset CSV must be placed where `src/dataset.py` expects it (see that file). Running any training script will create `results/scaler.pkl` and `results/target_stats.pkl` on first invocation.

## Running

All scripts are run as modules from `project/code/`:

### LSTM (window = 24)
```bat
python -m src.run_lstm A      :: configA — hidden=64
python -m src.run_lstm B      :: configB — hidden=128
python -m src.run_lstm all    :: both
```

### CNN (window selectable: 24 or 48)
```bat
python -m src.run_cnn A 24    :: configA  @ W=24
python -m src.run_cnn B 24    :: configB  @ W=24
python -m src.run_cnn A 48    :: configA  @ W=48
python -m src.run_cnn B 48    :: configB  @ W=48
python -m src.run_cnn all 48  :: both     @ W=48
```
Default window is 24 if omitted. Note: the evaluation script expects the W=48 outputs as the canonical `cnn_configA.json` / `cnn_configB.json`. To preserve W=24 ablation runs, rename them to `cnn_configA_w24.json` / `.pt` (and same for B) before the W=48 runs overwrite them.

### Hybrid (window = 48, fixed)
```bat
python -m src.run_hybrid A    :: 2 LSTM layers
python -m src.run_hybrid B    :: 3 LSTM layers
python -m src.run_hybrid all
```

### Evaluation (after all training is done)
```bat
python -m src.run_evaluation
```
Produces test/val RMSE & MAE for every checkpoint, comparison and ablation plots under `results/plots/`, and `results/evaluation_summary.json`.

## Results

Canonical numbers (from `archive/b2_master_table.csv`, single seed = 42, CPU training):

| Model  | Cfg | Window | Params  | Epochs | Train (s) | Test RMSE (°C) | Test MAE (°C) |
|--------|-----|--------|---------|--------|-----------|----------------|---------------|
| LSTM   | A   | 24     | 53 825  | 21     | 226.58    | 0.6964         | 0.5018        |
| LSTM   | B   | 24     | 205 953 | 29     | 599.31    | **0.6570**     | **0.4636**    |
| CNN    | A   | 24     | 7 649   | 12     | 64.90     | 1.5287         | 1.1726        |
| CNN    | B   | 24     | 29 377  | 13     | 84.15     | 1.0098         | 0.7850        |
| CNN    | A   | 48     | 7 649   | 17     | 87.39     | 0.9084         | 0.6911        |
| CNN    | B   | 48     | 29 377  | 26     | 215.36    | 0.8662         | 0.6607        |
| Hybrid | A   | 48     | 46 113  | 40     | 523.04    | **0.6636**     | **0.4701**    |
| Hybrid | B   | 48     | 54 561  | 16     | 216.29    | 0.7464         | 0.5546        |

Persistence baseline (predict `T(t+1) = T(t)`): RMSE ≈ **0.9631 °C**.

### Headline

- **LSTM_B** (RMSE **0.6570**) and **Hybrid_A** (RMSE **0.6636**) are statistically tied — the gap is well below seed-to-seed noise expected for a single run.
- **Hybrid_A wins on parameter efficiency**: 46 113 vs 205 953 params — **~4.5× fewer parameters** for equivalent accuracy.
- **CNN at W=24 fails the persistence baseline**: configA (1.5287) and configB (1.0098) are both worse than naïve persistence (0.9631). CNN only becomes competitive at W=48, and even then trails LSTM/Hybrid.

## Notes & Caveats

- **CPU-only training** (no CUDA used in reported runs); train times are wall-clock CPU.
- **Single seed (42)** per configuration — no variance estimates, so small RMSE gaps are not significant.
- **Mismatched windows across families** (intentional, per family-specific design):
  - LSTM: W=24
  - CNN: W=24 and W=48 (ablation)
  - Hybrid: W=48 only
  Cross-family comparisons should be read with this in mind.
- CNN's `best_val_loss` in the master table is in **unscaled units** (the CNN training scaled differently); LSTM/Hybrid `best_val_loss` is in scaled units. Use **test RMSE/MAE** for comparison, not val loss.
