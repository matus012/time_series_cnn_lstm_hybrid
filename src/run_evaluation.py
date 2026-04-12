"""Person 4 — Main evaluation script.

Loads all trained models, computes metrics, generates all plots,
and runs the window-size ablation study (24 vs 48).

Usage:
    python -m src.run_evaluation

Outputs (all saved to results/):
    - plots/loss_<model>.png        — loss curves per model
    - plots/predictions_<model>.png — last 100 predictions vs ground truth
    - plots/errors_<model>.png      — error distribution + scatter
    - plots/comparison.png          — RMSE/MAE/params across all models
    - plots/ablation_<model>.png    — window 24 vs 48 ablation
    - evaluation_summary.json       — all numeric results
"""

import json
from pathlib import Path

import joblib
import torch

from src.dataset import prepare_data
from src.evaluate import (
    compare_models,
    compute_metrics,
    plot_error_analysis,
    plot_loss_curves,
    plot_predictions,
)
from src.models.lstm import LSTMModel
from src.utils import get_device, set_seed

# ── Optional: import CNN / Hybrid when teammates deliver them ──────────────────
# from src.models.cnn import CNNModel
# from src.models.hybrid import HybridModel
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_target_stats() -> tuple[float, float]:
    stats = joblib.load(RESULTS_DIR / "target_stats.pkl")
    return stats["mean"], stats["std"]


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


def load_json_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_evaluation() -> None:
    set_seed(42)
    device = get_device()

    target_mean, target_std = load_target_stats()
    print(f"Target stats — mean: {target_mean:.3f} °C, std: {target_std:.3f} °C")

    # ── Load data for both window sizes ─────────────────────────────────────────
    loaders, target_col = prepare_data()
    train_loader_24, val_loader_24, test_loader_24 = loaders[24]
    train_loader_48, val_loader_48, test_loader_48 = loaders[48]

    # ── Define models to evaluate ───────────────────────────────────────────────
    # Each entry: (display_name, model_instance, checkpoint_path, results_json_path)
    model_configs = [
        (
            "LSTM_configA",
            LSTMModel(input_size=14, hidden_size=64, num_layers=2, dropout=0.2),
            str(RESULTS_DIR / "lstm_configA.pt"),
            str(RESULTS_DIR / "lstm_configA.json"),
        ),
        (
            "LSTM_configB",
            LSTMModel(input_size=14, hidden_size=128, num_layers=2, dropout=0.2),
            str(RESULTS_DIR / "lstm_configB.pt"),
            str(RESULTS_DIR / "lstm_configB.json"),
        ),
        # ── Uncomment when Person 2 delivers CNN ──────────────────────────────
        # ("CNN_configA", CNNModel(...), "results/cnn_configA.pt", "results/cnn_configA.json"),
        # ("CNN_configB", CNNModel(...), "results/cnn_configB.pt", "results/cnn_configB.json"),
        # ── Uncomment when Person 3 delivers Hybrid ───────────────────────────
        # ("Hybrid_configA", HybridModel(...), "results/hybrid_configA.pt", "results/hybrid_configA.json"),
        # ("Hybrid_configB", HybridModel(...), "results/hybrid_configB.pt", "results/hybrid_configB.json"),
    ]

    all_model_results: dict[str, dict] = {}

    for name, model, ckpt_path, json_path in model_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        # Load checkpoint
        if not Path(ckpt_path).exists():
            print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping.")
            continue

        model = load_model_checkpoint(model, ckpt_path, device)
        train_results = load_json_results(json_path)

        # Loss curves
        plot_loss_curves(
            train_results,
            model_name=name,
            save_path=str(PLOTS_DIR / f"loss_{name}.png"),
        )

        # Metrics on validation set
        val_metrics = compute_metrics(model, val_loader_24, target_mean, target_std, device)
        print(f"  Val  — RMSE: {val_metrics['rmse']:.4f} °C  |  MAE: {val_metrics['mae']:.4f} °C")

        # Metrics on test set
        test_metrics = compute_metrics(model, test_loader_24, target_mean, target_std, device)
        print(f"  Test — RMSE: {test_metrics['rmse']:.4f} °C  |  MAE: {test_metrics['mae']:.4f} °C")

        # Predictions plot (last 100 steps, test set)
        plot_predictions(
            test_metrics["predictions"],
            test_metrics["targets"],
            model_name=name,
            save_path=str(PLOTS_DIR / f"predictions_{name}.png"),
        )

        # Error analysis
        error_info = plot_error_analysis(
            test_metrics["predictions"],
            test_metrics["targets"],
            model_name=name,
            save_path=str(PLOTS_DIR / f"errors_{name}.png"),
        )

        all_model_results[name] = {
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "param_count": train_results["param_count"],
            "best_val_loss": train_results["best_val_loss"],
            "epoch_times": train_results["epoch_times"],
            "config": train_results.get("config", {}),
        }

    # ── Cross-model comparison ───────────────────────────────────────────────────
    if all_model_results:
        compare_models(all_model_results, save_path=str(PLOTS_DIR / "comparison.png"))

    # ── Ablation study: window size 24 vs 48 ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Ablation study: window size 24 vs 48")
    print(f"{'='*60}")

    ablation_results: dict[str, dict] = {}

    for name, model, ckpt_path, json_path in model_configs:
        if not Path(ckpt_path).exists():
            continue
        model = load_model_checkpoint(model, ckpt_path, device)

        for window_size, test_loader in [(24, test_loader_24), (48, test_loader_48)]:
            key = f"{name}_w{window_size}"
            metrics = compute_metrics(model, test_loader, target_mean, target_std, device)
            print(f"  {key:<35} RMSE: {metrics['rmse']:.4f} °C  MAE: {metrics['mae']:.4f} °C")
            ablation_results[key] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "window_size": window_size,
            }

    # ── Save summary JSON ────────────────────────────────────────────────────────
    summary = {
        "model_results": all_model_results,
        "ablation": ablation_results,
    }
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_evaluation()
