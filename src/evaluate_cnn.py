"""Evaluation script for CNN model on the test set."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from src.dataset import prepare_data
from src.models.cnn import CNNModel
from src.utils import get_device, set_seed


RESULTS_DIR = Path("results")


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target_stats(path: Path) -> dict:
    """Load saved target mean/std for inverse transform."""
    if not path.exists():
        raise FileNotFoundError(f"Missing target stats file: {path}")
    return joblib.load(path)


def build_model_from_config(config: dict) -> CNNModel:
    """Rebuild CNN model from saved config."""
    return CNNModel(
        input_size=14,
        channels_1=config["channels_1"],
        channels_2=config["channels_2"],
        kernel_size_1=config["kernel_size_1"],
        kernel_size_2=config["kernel_size_2"],
        dropout=config["dropout"],
    )


def load_model_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    """Load saved weights into model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict:
    """Evaluate model on test set and return metrics."""
    model.eval()

    preds_scaled = []
    targets_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)

            preds_scaled.append(outputs.cpu().numpy())
            targets_scaled.append(y_batch.cpu().numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0).reshape(-1)
    targets_scaled = np.concatenate(targets_scaled, axis=0).reshape(-1)

    mse_scaled = float(np.mean((preds_scaled - targets_scaled) ** 2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    mae_scaled = float(np.mean(np.abs(preds_scaled - targets_scaled)))

    preds_real = preds_scaled * target_std + target_mean
    targets_real = targets_scaled * target_std + target_mean

    mse_real = float(np.mean((preds_real - targets_real) ** 2))
    rmse_real = float(np.sqrt(mse_real))
    mae_real = float(np.mean(np.abs(preds_real - targets_real)))

    ss_res = float(np.sum((targets_real - preds_real) ** 2))
    ss_tot = float(np.sum((targets_real - np.mean(targets_real)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {
        "mse_scaled": mse_scaled,
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "mse_real": mse_real,
        "rmse_real": rmse_real,
        "mae_real": mae_real,
        "r2_real": r2,
        "num_test_samples": int(len(targets_real)),
    }


def evaluate_config(config_name: str, window_override: int | None = None) -> None:
    """Evaluate one CNN configuration."""
    json_path = RESULTS_DIR / f"cnn_config{config_name}.json"
    model_path = RESULTS_DIR / f"cnn_config{config_name}.pt"
    stats_path = RESULTS_DIR / "target_stats.pkl"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing config JSON: {json_path}")

    saved_results = load_json(json_path)
    config = saved_results["config"]

    if window_override is not None:
        config["window_size"] = window_override

    set_seed(42)
    device = get_device()

    all_loaders, target_col = prepare_data(batch_size=config["batch_size"])
    _, _, test_loader = all_loaders[config["window_size"]]

    target_stats = load_target_stats(stats_path)
    target_mean = float(target_stats["mean"])
    target_std = float(target_stats["std"])

    model = build_model_from_config(config).to(device)
    load_model_weights(model, model_path, device)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    output = {
        "model": f"cnn_config{config_name}",
        "target_column": target_col,
        "window_size": config["window_size"],
        "config": config,
        "metrics": metrics,
    }

    out_path = RESULTS_DIR / f"eval_cnn_config{config_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 60)
    print(f"CNN config{config_name} evaluation")
    print("=" * 60)
    print(f"Window size: {config['window_size']}")
    print(f"Test samples: {metrics['num_test_samples']}")
    print(f"Scaled MAE : {metrics['mae_scaled']:.6f}")
    print(f"Scaled RMSE: {metrics['rmse_scaled']:.6f}")
    print(f"Real MAE   : {metrics['mae_real']:.6f} °C")
    print(f"Real RMSE  : {metrics['rmse_real']:.6f} °C")
    print(f"R²         : {metrics['r2_real']:.6f}")
    print(f"Saved to   : {out_path}")


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    window_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if config_arg in ("A", "all"):
        evaluate_config("A", window_arg)

    if config_arg in ("B", "all"):
        evaluate_config("B", window_arg)