"""Evaluation metrics and plotting for model comparison. (Person 4)"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> dict:
    """Compute evaluation metrics on a dataset.

    Inverse-transforms predictions and targets using:
        original = scaled * target_std + target_mean

    Load target_stats via: joblib.load('results/target_stats.pkl')
    Then pass target_stats['mean'] and target_stats['std'].

    Args:
        model: Trained model, already on device.
        dataloader: Test DataLoader.
        target_mean: Mean of target column from training set scaler.
        target_std: Std of target column from training set scaler.
        device: torch.device.

    Returns:
        Dict with keys:
            - rmse: float (on original scale, degC)
            - mae: float (on original scale, degC)
            - predictions: np.ndarray (original scale)
            - targets: np.ndarray (original scale)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)  # (batch, 1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_preds, axis=0).squeeze()   # (N,)
    targets = np.concatenate(all_targets, axis=0).squeeze()      # (N,)

    # Inverse transform to original scale (degC)
    predictions = predictions * target_std + target_mean
    targets = targets * target_std + target_mean

    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mae = float(np.mean(np.abs(predictions - targets)))

    return {
        "rmse": rmse,
        "mae": mae,
        "predictions": predictions,
        "targets": targets,
    }


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot predicted vs actual temperature values (last 100 steps).

    Args:
        predictions: Model predictions (original scale).
        targets: Ground truth (original scale).
        model_name: Name for plot title.
        save_path: If provided, save figure to this path.
    """
    n = min(100, len(predictions))
    preds_plot = predictions[-n:]
    targets_plot = targets[-n:]
    steps = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, targets_plot, label="Ground Truth", color="steelblue", linewidth=1.5)
    ax.plot(steps, preds_plot, label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    ax.set_title(f"{model_name} — Predictions vs Ground Truth (last {n} steps)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved predictions plot to {save_path}")

    plt.close(fig)


def plot_loss_curves(
    results: dict,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        results: Dict from train_model() containing train_losses and val_losses.
        model_name: Name for plot title.
        save_path: If provided, save figure to this path.
    """
    train_losses = results["train_losses"]
    val_losses = results["val_losses"]
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", color="tomato", linewidth=1.5)
    ax.set_title(f"{model_name} — Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved loss curve to {save_path}")

    plt.close(fig)


def plot_error_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    top_n: int = 20,
    save_path: str | None = None,
) -> dict:
    """Identify and plot the largest prediction errors.

    Args:
        predictions: Model predictions (original scale).
        targets: Ground truth (original scale).
        model_name: Name for plot title.
        top_n: Number of largest errors to highlight.
        save_path: If provided, save figure to this path.

    Returns:
        Dict with 'top_error_indices', 'top_errors', 'top_predictions', 'top_targets'.
    """
    errors = np.abs(predictions - targets)
    top_indices = np.argsort(errors)[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Error distribution
    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"{model_name} — Error Distribution")
    axes[0].set_xlabel("|Prediction - Target| (°C)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    # Top errors scatter
    axes[1].scatter(targets, predictions, alpha=0.2, s=5, color="steelblue", label="All predictions")
    axes[1].scatter(
        targets[top_indices], predictions[top_indices],
        alpha=0.8, s=30, color="tomato", label=f"Top {top_n} errors", zorder=5
    )
    lim_min = min(targets.min(), predictions.min()) - 1
    lim_max = max(targets.max(), predictions.max()) + 1
    axes[1].plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Perfect prediction")
    axes[1].set_title(f"{model_name} — Predictions vs Targets")
    axes[1].set_xlabel("Target (°C)")
    axes[1].set_ylabel("Predicted (°C)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved error analysis to {save_path}")

    plt.close(fig)

    return {
        "top_error_indices": top_indices.tolist(),
        "top_errors": errors[top_indices].tolist(),
        "top_predictions": predictions[top_indices].tolist(),
        "top_targets": targets[top_indices].tolist(),
    }


def compare_models(
    model_results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Generate comparison table and bar chart across all models.

    Args:
        model_results: {model_name: {"rmse": float, "mae": float,
                        "param_count": int, "best_val_loss": float,
                        "total_train_time": float (seconds), ...}}
        save_path: If provided, save comparison figure to this path.
    """
    names = list(model_results.keys())
    rmses = [model_results[n].get("rmse", float("nan")) for n in names]
    maes = [model_results[n].get("mae", float("nan")) for n in names]
    params = [model_results[n].get("param_count", 0) for n in names]
    train_times = [
        sum(model_results[n].get("epoch_times", [0])) for n in names
    ]

    # Print comparison table
    header = f"{'Model':<30} {'RMSE (°C)':>10} {'MAE (°C)':>10} {'Params':>10} {'Train time (s)':>15}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, rmse, mae, p, t in zip(names, rmses, maes, params, train_times):
        print(f"{name:<30} {rmse:>10.4f} {mae:>10.4f} {p:>10,} {t:>15.1f}")
    print("=" * len(header) + "\n")

    # Bar chart: RMSE and MAE side by side
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = axes[0].bar(x - width / 2, rmses, width, label="RMSE", color="steelblue", alpha=0.85)
    bars2 = axes[0].bar(x + width / 2, maes, width, label="MAE", color="tomato", alpha=0.85)
    axes[0].set_title("RMSE & MAE by Model (°C)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Error (°C)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Parameter count
    axes[1].bar(x, params, color="mediumseagreen", alpha=0.85)
    axes[1].set_title("Parameter Count by Model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Parameters")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, p in enumerate(params):
        axes[1].text(i, p + max(params) * 0.01, f"{p:,}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved comparison chart to {save_path}")

    plt.close(fig)
