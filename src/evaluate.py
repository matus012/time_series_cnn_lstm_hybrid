"""Evaluation metrics and plotting for model comparison. (Person 4)"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> dict:
    """Compute RMSE and MAE on a dataset, returning results on the original scale.

    Inverse-transforms predictions and targets using:
        original = scaled * target_std + target_mean

    Load target_stats via: joblib.load('results/target_stats.pkl')
    Then pass target_stats['mean'] and target_stats['std'].

    Args:
        model:        Trained model already on device.
        dataloader:   DataLoader (val or test).
        target_mean:  Mean of target column from training scaler.
        target_std:   Std  of target column from training scaler.
        device:       torch.device.

    Returns:
        Dict with keys:
            rmse        — float, degrees C (original scale)
            mae         — float, degrees C (original scale)
            predictions — np.ndarray, shape (N,), original scale
            targets     — np.ndarray, shape (N,), original scale
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)                      # (batch, 1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_preds,   axis=0).squeeze()   # (N,)
    targets     = np.concatenate(all_targets, axis=0).squeeze()   # (N,)

    # Inverse-transform to degrees C
    predictions = predictions * target_std + target_mean
    targets     = targets     * target_std + target_mean

    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mae  = float(np.mean(np.abs(predictions - targets)))

    return {"rmse": rmse, "mae": mae, "predictions": predictions, "targets": targets}


def load_precomputed_metrics(eval_json_path: str) -> dict:
    """Load test metrics from a pre-computed eval JSON (e.g. from Person 2).

    Supports both formats:
      • {"metrics": {"rmse_real": ..., "mae_real": ...}}   (Person 2's format)
      • {"rmse": ..., "mae": ...}                          (our format)

    Returns:
        Dict with at minimum keys: rmse (float), mae (float).
        predictions / targets will be absent (no raw arrays saved in JSON).
    """
    with open(eval_json_path) as f:
        data = json.load(f)

    # Person 2's evaluate_cnn.py stores metrics one level deeper
    if "metrics" in data:
        m = data["metrics"]
        return {
            "rmse": float(m.get("rmse_real", m.get("rmse", float("nan")))),
            "mae":  float(m.get("mae_real",  m.get("mae",  float("nan")))),
        }

    # Our own format
    return {
        "rmse": float(data.get("rmse", float("nan"))),
        "mae":  float(data.get("mae",  float("nan"))),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot the last 100 predicted vs actual temperature values.

    Args:
        predictions: Model predictions (original scale, degrees C).
        targets:     Ground truth     (original scale, degrees C).
        model_name:  Name used in the plot title.
        save_path:   If given, save figure here; otherwise show interactively.
    """
    n = min(100, len(predictions))
    steps = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, targets[-n:],     label="Ground Truth", color="steelblue", linewidth=1.5)
    ax.plot(steps, predictions[-n:], label="Predicted",    color="tomato",    linewidth=1.5, linestyle="--")
    ax.set_title(f"{model_name} — Predictions vs Ground Truth (last {n} steps)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_loss_curves(
    results: dict,
    model_name: str,
    save_path: str | None = None,
    skip_val: bool = False,
) -> None:
    """Plot training (and optionally validation) loss curves from a results dict.

    Args:
        results:    Dict from train_model() with train_losses / val_losses keys.
        model_name: Name used in the plot title.
        save_path:  If given, save figure here; otherwise show interactively.
        skip_val:   If True, only plot train loss (use when val_losses are known
                    to be unreliable, e.g. the CNN anomalous training run).
    """
    train_losses = results.get("train_losses", [])
    val_losses   = results.get("val_losses",   [])
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue", linewidth=1.5)

    if not skip_val and val_losses:
        ax.plot(epochs, val_losses, label="Val Loss", color="tomato", linewidth=1.5)
    elif skip_val:
        ax.set_title(f"{model_name} — Training Loss  (val loss omitted — anomalous run)")
    ax.set_title(ax.get_title() or f"{model_name} — Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


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
        targets:     Ground truth (original scale).
        model_name:  Name used in the plot title.
        top_n:       Number of worst errors to highlight.
        save_path:   If given, save figure here.

    Returns:
        Dict with top_error_indices, top_errors, top_predictions, top_targets.
    """
    errors     = np.abs(predictions - targets)
    top_idx    = np.argsort(errors)[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Error distribution
    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"{model_name} — Error Distribution")
    axes[0].set_xlabel("|Prediction − Target| (°C)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    # Scatter: all predictions vs targets, top errors highlighted
    lim = [min(targets.min(), predictions.min()) - 1,
           max(targets.max(), predictions.max()) + 1]
    axes[1].scatter(targets, predictions, alpha=0.15, s=4,  color="steelblue", label="All")
    axes[1].scatter(targets[top_idx], predictions[top_idx],
                    alpha=0.8, s=30, color="tomato", zorder=5,
                    label=f"Top-{top_n} errors")
    axes[1].plot(lim, lim, "k--", linewidth=1, label="Perfect prediction")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_title(f"{model_name} — Predicted vs Target")
    axes[1].set_xlabel("Target (°C)"); axes[1].set_ylabel("Predicted (°C)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return {
        "top_error_indices":  top_idx.tolist(),
        "top_errors":         errors[top_idx].tolist(),
        "top_predictions":    predictions[top_idx].tolist(),
        "top_targets":        targets[top_idx].tolist(),
    }


def compare_models(
    model_results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Print a comparison table and save bar charts for all models.

    Args:
        model_results: {
            model_name: {
                "rmse":          float,
                "mae":           float,
                "param_count":   int,
                "epoch_times":   list[float],
                ...
            }
        }
        save_path: If given, save the figure here.
    """
    names       = list(model_results.keys())
    rmses       = [model_results[n].get("rmse",        float("nan")) for n in names]
    maes        = [model_results[n].get("mae",         float("nan")) for n in names]
    params      = [model_results[n].get("param_count", 0)            for n in names]
    train_times = [sum(model_results[n].get("epoch_times", [0]))     for n in names]

    # Console table
    col = 28
    hdr = f"{'Model':<{col}} {'RMSE (°C)':>10} {'MAE (°C)':>10} {'Params':>10} {'Train time (s)':>15}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for n, r, m, p, t in zip(names, rmses, maes, params, train_times):
        r_s = f"{r:.4f}" if not np.isnan(r) else "  n/a  "
        m_s = f"{m:.4f}" if not np.isnan(m) else "  n/a  "
        print(f"{n:<{col}} {r_s:>10} {m_s:>10} {p:>10,} {t:>15.1f}")
    print("=" * len(hdr) + "\n")

    x     = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE / MAE bars
    b1 = axes[0].bar(x - width / 2, rmses, width, label="RMSE", color="steelblue", alpha=0.85)
    b2 = axes[0].bar(x + width / 2, maes,  width, label="MAE",  color="tomato",    alpha=0.85)
    axes[0].set_title("Test RMSE & MAE by Model (°C)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Error (°C)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar in (*b1, *b2):
        h = bar.get_height()
        if not np.isnan(h):
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Parameter counts
    axes[1].bar(x, params, color="mediumseagreen", alpha=0.85)
    axes[1].set_title("Parameter Count by Model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Parameters")
    axes[1].grid(True, axis="y", alpha=0.3)
    mx = max(params) if params else 1
    for i, p in enumerate(params):
        axes[1].text(i, p + mx * 0.01, f"{p:,}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_ablation(
    model_name: str,
    ablation_row: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Bar chart comparing window=24 vs window=48 for a single model.

    Args:
        model_name:   Name used in the title.
        ablation_row: {"w24": {"rmse": float, "mae": float},
                       "w48": {"rmse": float, "mae": float}}
        save_path:    If given, save figure here.
    """
    labels = ["Window = 24", "Window = 48"]
    rmses  = [ablation_row.get("w24", {}).get("rmse", float("nan")),
              ablation_row.get("w48", {}).get("rmse", float("nan"))]
    maes   = [ablation_row.get("w24", {}).get("mae",  float("nan")),
              ablation_row.get("w48", {}).get("mae",  float("nan"))]

    x     = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, rmses, width, label="RMSE", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, maes,  width, label="MAE",  color="tomato",    alpha=0.85)
    ax.set_title(f"{model_name} — Ablation: Window Size 24 vs 48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (°C)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ── Internal helper ───────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to disk if a path is given, then close it."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
