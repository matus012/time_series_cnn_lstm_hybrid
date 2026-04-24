"""Person 4 — Main evaluation script.

Loads all trained models, computes metrics, generates all plots,
and runs the window-size ablation study (24 vs 48).

Usage (from project root):
    python -m src.run_evaluation

Behaviour:
  • Loss-curve plots are always generated from training JSONs (no dataset needed).
  • If the dataset CSV + a model checkpoint (.pt) exist → full inference:
      predictions-vs-ground-truth plot, error analysis, RMSE/MAE.
  • If a checkpoint is missing → falls back to any pre-computed eval JSON
      (e.g. results/eval_cnn_configA.json written by Person 2).
  • Everything missing is reported clearly; nothing crashes silently.

Outputs saved to results/plots/:
    loss_<model>.png          — training & validation loss curves
    predictions_<model>.png   — last 100 predictions vs ground truth  (needs checkpoint)
    errors_<model>.png        — error distribution + scatter           (needs checkpoint)
    comparison.png            — RMSE / MAE / params across all models
    ablation_<model>.png      — window 24 vs 48 bar chart              (needs checkpoint)
    results/evaluation_summary.json — all numeric results
"""

import json
from pathlib import Path

import torch

from src.evaluate import (
    compare_models,
    compute_metrics,
    load_precomputed_metrics,
    plot_ablation,
    plot_error_analysis,
    plot_loss_curves,
    plot_predictions,
)
from src.models.cnn import CNNModel
from src.models.hybrid import HybridModel
from src.models.lstm import LSTMModel
from src.utils import get_device, set_seed

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent   # project root
RESULTS_DIR  = ROOT / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"
DATA_CSV     = ROOT / "data" / "jena_climate_2009_2016.csv"
STATS_PKL    = RESULTS_DIR / "target_stats.pkl"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model registry ────────────────────────────────────────────────────────────
# Defines every trained configuration in one place.
#   checkpoint      : path to .pt file (may not exist yet)
#   training_json   : path to training-results JSON (always exists after training)
#   eval_json       : optional pre-computed test-metrics JSON (Person 2 provided these)
#   window_size     : must match the window used during training
#   skip_val_loss   : True when the recorded val_losses are known-bad (CNN anomaly)
MODEL_REGISTRY = [
    {
        "name":          "LSTM_configA",
        "model_fn":      lambda: LSTMModel(input_size=14, hidden_size=64,  num_layers=2, dropout=0.2),
        "checkpoint":    RESULTS_DIR / "lstm_configA.pt",
        "training_json": RESULTS_DIR / "lstm_configA.json",
        "eval_json":     None,
        "window_size":   24,
        "skip_val_loss": False,
    },
    {
        "name":          "LSTM_configB",
        "model_fn":      lambda: LSTMModel(input_size=14, hidden_size=128, num_layers=2, dropout=0.2),
        "checkpoint":    RESULTS_DIR / "lstm_configB.pt",
        "training_json": RESULTS_DIR / "lstm_configB.json",
        "eval_json":     None,
        "window_size":   24,
        "skip_val_loss": False,
    },
    {
        "name":          "CNN_configA",
        "model_fn":      lambda: CNNModel(input_size=14, channels_1=32,  channels_2=64,  kernel_size_1=3, kernel_size_2=3, dropout=0.2),
        # Person 2 placed their checkpoints inside their own sub-folder
        "checkpoint":    ROOT / "Person 2 - CNN" / "results" / "cnn_configA.pt",
        "training_json": RESULTS_DIR / "cnn_configA.json",
        "eval_json":     RESULTS_DIR / "eval_cnn_configA.json",
        "window_size":   24,
        # CNN training JSONs show val_losses of 22-120 (scaler mismatch during
        # Person 2's run); skip plotting the broken val curve.
        "skip_val_loss": True,
    },
    {
        "name":          "CNN_configB",
        "model_fn":      lambda: CNNModel(input_size=14, channels_1=64,  channels_2=128, kernel_size_1=5, kernel_size_2=3, dropout=0.3),
        "checkpoint":    ROOT / "Person 2 - CNN" / "results" / "cnn_configB.pt",
        "training_json": RESULTS_DIR / "cnn_configB.json",
        "eval_json":     RESULTS_DIR / "eval_cnn_configB.json",
        "window_size":   24,
        "skip_val_loss": True,
    },
    {
        "name":          "Hybrid_configA",
        "model_fn":      lambda: HybridModel(input_size=14, cnn_channels=64, kernel_size=5, lstm_hidden_size=32, lstm_num_layers=2, dropout=0.2),
        "checkpoint":    RESULTS_DIR / "hybrid_configA.pt",
        "training_json": RESULTS_DIR / "hybrid_configA.json",
        "eval_json":     None,
        "window_size":   48,   # Hybrid was trained on window=48 (MaxPool halves it for LSTM)
        "skip_val_loss": False,
    },
    {
        "name":          "Hybrid_configB",
        "model_fn":      lambda: HybridModel(input_size=14, cnn_channels=64, kernel_size=5, lstm_hidden_size=32, lstm_num_layers=3, dropout=0.2),
        "checkpoint":    RESULTS_DIR / "hybrid_configB.pt",
        "training_json": RESULTS_DIR / "hybrid_configB.json",
        "eval_json":     None,
        "window_size":   48,
        "skip_val_loss": False,
    },
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_target_stats() -> tuple[float, float] | None:
    """Return (mean, std) from saved pkl, or None if not yet generated."""
    if not STATS_PKL.exists():
        return None
    import joblib
    s = joblib.load(STATS_PKL)
    return float(s["mean"]), float(s["std"])


def _get_loaders() -> dict | None:
    """Build DataLoaders for both window sizes; return None if dataset missing."""
    if not DATA_CSV.exists():
        return None
    from src.dataset import prepare_data
    set_seed(42)
    loaders, _ = prepare_data(csv_path=str(DATA_CSV))
    return loaders     # {24: (train, val, test), 48: (train, val, test)}


def _load_checkpoint(entry: dict, device: torch.device) -> torch.nn.Module | None:
    ckpt = Path(entry["checkpoint"])
    if not ckpt.exists():
        return None
    model: torch.nn.Module = entry["model_fn"]()
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    set_seed(42)
    device = get_device()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    # _get_loaders() calls prepare_data() which writes target_stats.pkl,
    # so we must load loaders FIRST and read the pkl file afterwards.
    loaders    = _get_loaders()
    dataset_ok = loaders is not None

    if not dataset_ok:
        print(
            "\n[WARNING] Dataset not found at data/jena_climate_2009_2016.csv\n"
            "  Model inference and ablation study will be skipped.\n"
            "  Loss curves and any pre-computed metrics will still be generated.\n"
            "  Download from: https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016\n"
        )
        target_mean, target_std = 0.0, 1.0   # unused, but keeps later code clean
    else:
        target_stats = _load_target_stats()  # pkl now guaranteed to exist
        assert target_stats is not None
        target_mean, target_std = target_stats
        print(f"\nTarget stats  mean={target_mean:.3f} °C  std={target_std:.3f} °C")

    all_results: dict[str, dict] = {}

    # ── Per-model evaluation ──────────────────────────────────────────────────
    for entry in MODEL_REGISTRY:
        name         = entry["name"]
        window       = entry["window_size"]
        skip_val     = entry["skip_val_loss"]
        train_json_p = Path(entry["training_json"])
        eval_json_p  = Path(entry["eval_json"]) if entry["eval_json"] else None

        print(f"\n{'='*62}")
        print(f"  {name}  (window_size={window})")
        print(f"{'='*62}")

        # 1. Loss curves ── always possible when training JSON exists ──────────
        param_count  = 0
        epoch_times  = []
        best_val     = float("nan")

        if train_json_p.exists():
            train_res   = _load_json(train_json_p)
            param_count = train_res.get("param_count", 0)
            epoch_times = train_res.get("epoch_times", [])
            best_val    = train_res.get("best_val_loss", float("nan"))

            plot_loss_curves(
                train_res,
                model_name=name,
                save_path=str(PLOTS_DIR / f"loss_{name}.png"),
                skip_val=skip_val,
            )
        else:
            print(f"  [WARNING] Training JSON not found: {train_json_p}")

        # 2. Test metrics ── try live inference first ──────────────────────────
        test_metrics: dict | None = None

        if dataset_ok:
            model = _load_checkpoint(entry, device)
            if model is not None:
                _, _, test_loader = loaders[window]
                test_metrics = compute_metrics(
                    model, test_loader, target_mean, target_std, device
                )
                print(
                    f"  [Inference]  RMSE={test_metrics['rmse']:.4f} °C  "
                    f"MAE={test_metrics['mae']:.4f} °C"
                )
                plot_predictions(
                    test_metrics["predictions"],
                    test_metrics["targets"],
                    model_name=name,
                    save_path=str(PLOTS_DIR / f"predictions_{name}.png"),
                )
                plot_error_analysis(
                    test_metrics["predictions"],
                    test_metrics["targets"],
                    model_name=name,
                    save_path=str(PLOTS_DIR / f"errors_{name}.png"),
                )
            else:
                print(f"  [WARNING] Checkpoint missing: {entry['checkpoint']}")

        # 3. Fall back to pre-computed eval JSON ──────────────────────────────
        if test_metrics is None and eval_json_p and eval_json_p.exists():
            test_metrics = load_precomputed_metrics(str(eval_json_p))
            print(
                f"  [Pre-computed]  RMSE={test_metrics['rmse']:.4f} °C  "
                f"MAE={test_metrics['mae']:.4f} °C"
            )

        if test_metrics is None:
            print("  [INFO] No test metrics available for this model.")

        # 4. Accumulate for comparison table ──────────────────────────────────
        all_results[name] = {
            "rmse":          test_metrics["rmse"] if test_metrics else float("nan"),
            "mae":           test_metrics["mae"]  if test_metrics else float("nan"),
            "param_count":   param_count,
            "best_val_loss": best_val if not skip_val else float("nan"),
            "epoch_times":   epoch_times,
            "window_size":   window,
        }

    # ── Cross-model comparison ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Cross-model comparison")
    print(f"{'='*62}")
    compare_models(all_results, save_path=str(PLOTS_DIR / "comparison.png"))

    # ── Ablation: window 24 vs 48 ─────────────────────────────────────────────
    ablation_results: dict[str, dict] = {}

    if dataset_ok:
        print(f"\n{'='*62}")
        print("  Ablation study — window size 24 vs 48")
        print(f"{'='*62}")
        for entry in MODEL_REGISTRY:
            name = entry["name"]
            ckpt = Path(entry["checkpoint"])
            if not ckpt.exists():
                print(f"  Skipping {name} — checkpoint not available.")
                continue
            model = _load_checkpoint(entry, device)
            row: dict[str, dict] = {}
            for ws in (24, 48):
                _, _, tl = loaders[ws]
                m = compute_metrics(model, tl, target_mean, target_std, device)
                row[f"w{ws}"] = {"rmse": m["rmse"], "mae": m["mae"]}
                print(f"  {name:<28} window={ws}  RMSE={m['rmse']:.4f}°C  MAE={m['mae']:.4f}°C")
            if len(row) == 2:
                plot_ablation(name, row, save_path=str(PLOTS_DIR / f"ablation_{name}.png"))
                ablation_results[name] = row
    else:
        print("\n  [INFO] Ablation skipped — dataset not available.")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"model_results": all_results, "ablation": ablation_results},
            f, indent=2, default=str,
        )
    print(f"\n  Summary  -> {summary_path}")
    print(f"  Plots    -> {PLOTS_DIR}\n")


if __name__ == "__main__":
    run_evaluation()
