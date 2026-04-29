"""Main evaluation script for all available models.

This script:
- loads all available trained model checkpoints
- uses the correct window size from each model's saved config
- computes validation and test metrics
- generates loss/prediction/error plots
- creates a shared comparison summary
- creates window-size ablation summaries
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import torch

from src.dataset import prepare_data
from src.evaluate import (
    compare_models,
    compute_metrics,
    plot_ablation,
    plot_error_analysis,
    plot_loss_curves,
    plot_predictions,
)
from src.models.cnn import CNNModel
from src.models.hybrid import HybridModel
from src.models.lstm import LSTMModel
from src.utils import get_device, set_seed

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_target_stats() -> tuple[float, float]:
    """Load target mean/std used for inverse transform."""
    stats_path = RESULTS_DIR / "target_stats.pkl"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing target stats file: {stats_path}. "
            "Run prepare_data() first so scaler and target stats are created."
        )
    stats = joblib.load(stats_path)
    return float(stats["mean"]), float(stats["std"])


def load_json_results(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load checkpoint into model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return model


def build_model(model_type: str, config: dict) -> torch.nn.Module:
    """Build model instance from config metadata."""
    if model_type == "cnn":
        return CNNModel(
            input_size=14,
            channels_1=config["channels_1"],
            channels_2=config["channels_2"],
            kernel_size_1=config["kernel_size_1"],
            kernel_size_2=config["kernel_size_2"],
            dropout=config["dropout"],
        )

    if model_type == "lstm":
        return LSTMModel(
            input_size=14,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

    if model_type == "hybrid":
        return HybridModel(
            input_size=14,
            cnn_channels=config["cnn_channels"],
            kernel_size=config["kernel_size"],
            lstm_hidden_size=config["lstm_hidden_size"],
            lstm_num_layers=config["lstm_num_layers"],
            dropout=config["dropout"],
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def plot_window_ablation(ablation_summary: dict, save_path: str | None = None) -> None:
    """Plot RMSE by window size for each model family if available."""
    families = []
    rmse_24 = []
    rmse_48 = []

    for family, family_data in ablation_summary.items():
        if not family_data:
            continue
        families.append(family.upper())
        rmse_24.append(family_data.get("24", {}).get("rmse", float("nan")))
        rmse_48.append(family_data.get("48", {}).get("rmse", float("nan")))

    if not families:
        return

    x = range(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar([i - width / 2 for i in x], rmse_24, width, label="Window 24")
    bars2 = ax.bar([i + width / 2 for i in x], rmse_48, width, label="Window 48")

    ax.set_title("Window Size Ablation (RMSE)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h == h:  # not NaN
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved ablation plot to {save_path}")

    plt.close(fig)


def evaluate_single_model(
    display_name: str,
    model_type: str,
    train_json_path: Path,
    checkpoint_path: Path,
    loaders: dict,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> dict | None:
    """Evaluate one trained model using its own saved config."""
    if not train_json_path.exists():
        print(f"WARNING: missing training JSON: {train_json_path}")
        return None

    if not checkpoint_path.exists():
        print(f"WARNING: missing checkpoint: {checkpoint_path}")
        return None

    train_results = load_json_results(train_json_path)
    config = train_results.get("config", {})
    window_size = int(config.get("window_size", 24))

    if window_size not in loaders:
        print(f"WARNING: missing loaders for window size {window_size}")
        return None

    _, val_loader, test_loader = loaders[window_size]

    model = build_model(model_type, config)
    model = load_model_checkpoint(model, checkpoint_path, device)

    print(f"\n{'=' * 60}")
    print(f"Evaluating {display_name} | window={window_size}")
    print(f"{'=' * 60}")

    val_metrics = compute_metrics(model, val_loader, target_mean, target_std, device)
    test_metrics = compute_metrics(model, test_loader, target_mean, target_std, device)

    print(f"Val  - RMSE: {val_metrics['rmse']:.4f} °C | MAE: {val_metrics['mae']:.4f} °C")
    print(f"Test - RMSE: {test_metrics['rmse']:.4f} °C | MAE: {test_metrics['mae']:.4f} °C")

    plot_loss_curves(
        train_results,
        model_name=display_name,
        save_path=str(PLOTS_DIR / f"loss_{display_name}.png"),
    )
    plot_predictions(
        test_metrics["predictions"],
        test_metrics["targets"],
        model_name=display_name,
        save_path=str(PLOTS_DIR / f"predictions_{display_name}.png"),
    )
    error_info = plot_error_analysis(
        test_metrics["predictions"],
        test_metrics["targets"],
        model_name=display_name,
        save_path=str(PLOTS_DIR / f"errors_{display_name}.png"),
    )

    return {
        "model_type": model_type,
        "window_size": window_size,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "param_count": train_results.get("param_count", 0),
        "best_val_loss": train_results.get("best_val_loss", None),
        "epoch_times": train_results.get("epoch_times", []),
        "config": config,
        "error_info": error_info,
    }


def build_family_ablation_summary(all_results: dict[str, dict]) -> dict[str, dict]:
    """Group best results by model family and window size."""
    ablation_summary: dict[str, dict] = {
        "cnn": {},
        "lstm": {},
        "hybrid": {},
    }

    for model_name, result in all_results.items():
        family = result["model_type"]
        window_str = str(result["window_size"])

        current = ablation_summary[family].get(window_str)
        if current is None or result["rmse"] < current["rmse"]:
            ablation_summary[family][window_str] = {
                "model_name": model_name,
                "rmse": result["rmse"],
                "mae": result["mae"],
            }

    return ablation_summary


def build_cnn_config_ablation_summary(all_results: dict[str, dict]) -> dict[str, dict]:
    """Build per-config CNN ablation summary for configA and configB."""
    cnn_ablation = {
        "CNN_configA": {},
        "CNN_configB": {},
    }

    for model_name, result in all_results.items():
        if result["model_type"] != "cnn":
            continue

        window_key = f"w{result['window_size']}"

        if model_name.startswith("CNN_configA"):
            cnn_ablation["CNN_configA"][window_key] = {
                "rmse": result["rmse"],
                "mae": result["mae"],
                "model_name": model_name,
            }

        if model_name.startswith("CNN_configB"):
            cnn_ablation["CNN_configB"][window_key] = {
                "rmse": result["rmse"],
                "mae": result["mae"],
                "model_name": model_name,
            }

    return cnn_ablation


def run_evaluation() -> None:
    """Run shared evaluation for all available models."""
    set_seed(42)
    device = get_device()

    # Robust order:
    # 1. prepare_data() creates scaler/target_stats if missing
    # 2. then load target statistics
    loaders, _ = prepare_data()
    target_mean, target_std = load_target_stats()
    print(f"Target stats - mean: {target_mean:.3f} °C, std: {target_std:.3f} °C")

    # Core models used in the main final comparison
    core_model_entries = [
        ("LSTM_configA", "lstm", RESULTS_DIR / "lstm_configA.json", RESULTS_DIR / "lstm_configA.pt"),
        ("LSTM_configB", "lstm", RESULTS_DIR / "lstm_configB.json", RESULTS_DIR / "lstm_configB.pt"),
        ("CNN_configA", "cnn", RESULTS_DIR / "cnn_configA.json", RESULTS_DIR / "cnn_configA.pt"),
        ("CNN_configB", "cnn", RESULTS_DIR / "cnn_configB.json", RESULTS_DIR / "cnn_configB.pt"),
        ("Hybrid_configA", "hybrid", RESULTS_DIR / "hybrid_configA.json", RESULTS_DIR / "hybrid_configA.pt"),
        ("Hybrid_configB", "hybrid", RESULTS_DIR / "hybrid_configB.json", RESULTS_DIR / "hybrid_configB.pt"),
    ]

    # Optional extra runs for ablation / archived experiments
    extra_model_entries = [
        ("CNN_configA_w24", "cnn", RESULTS_DIR / "cnn_configA_w24.json", RESULTS_DIR / "cnn_configA_w24.pt"),
        ("CNN_configB_w24", "cnn", RESULTS_DIR / "cnn_configB_w24.json", RESULTS_DIR / "cnn_configB_w24.pt"),
    ]

    core_results: dict[str, dict] = {}
    extra_results: dict[str, dict] = {}

    for display_name, model_type, json_path, ckpt_path in core_model_entries:
        result = evaluate_single_model(
            display_name=display_name,
            model_type=model_type,
            train_json_path=json_path,
            checkpoint_path=ckpt_path,
            loaders=loaders,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )
        if result is not None:
            core_results[display_name] = result

    for display_name, model_type, json_path, ckpt_path in extra_model_entries:
        result = evaluate_single_model(
            display_name=display_name,
            model_type=model_type,
            train_json_path=json_path,
            checkpoint_path=ckpt_path,
            loaders=loaders,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )
        if result is not None:
            extra_results[display_name] = result

    # Main comparison only for final core models
    if core_results:
        compare_models(
            core_results,
            save_path=str(PLOTS_DIR / "comparison.png"),
        )

    # Ablation uses both core and extra results
    all_results = {**core_results, **extra_results}

    family_ablation_summary = build_family_ablation_summary(all_results)
    plot_window_ablation(
        family_ablation_summary,
        save_path=str(PLOTS_DIR / "ablation_windows.png"),
    )

    # CNN-specific ablation plots for configA and configB
    cnn_config_ablation = build_cnn_config_ablation_summary(all_results)

    if "w24" in cnn_config_ablation["CNN_configA"] and "w48" in cnn_config_ablation["CNN_configA"]:
        plot_ablation(
            model_name="CNN_configA",
            ablation_row=cnn_config_ablation["CNN_configA"],
            save_path=str(PLOTS_DIR / "ablation_CNN_configA.png"),
        )

    if "w24" in cnn_config_ablation["CNN_configB"] and "w48" in cnn_config_ablation["CNN_configB"]:
        plot_ablation(
            model_name="CNN_configB",
            ablation_row=cnn_config_ablation["CNN_configB"],
            save_path=str(PLOTS_DIR / "ablation_CNN_configB.png"),
        )

    summary = {
        "model_results": core_results,
        "extra_results": extra_results,
        "ablation_by_family": family_ablation_summary,
        "ablation_by_config": cnn_config_ablation,
    }

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_evaluation()