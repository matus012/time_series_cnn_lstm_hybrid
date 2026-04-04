"""Evaluation metrics and plotting for model comparison. (Person 4 implements)"""

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
    # TODO: Person 4 implements
    # 1. model.eval() + torch.no_grad()
    # 2. Iterate dataloader, collect predictions and targets
    # 3. Inverse transform: val = val * target_std + target_mean
    # 4. Compute RMSE = sqrt(mean((pred - target)^2))
    # 5. Compute MAE = mean(|pred - target|)
    raise NotImplementedError("Person 4: implement this function")


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot predicted vs actual temperature values.

    Args:
        predictions: Model predictions (original scale).
        targets: Ground truth (original scale).
        model_name: Name for plot title.
        save_path: If provided, save figure to this path.
    """
    # TODO: Person 4 implements
    raise NotImplementedError("Person 4: implement this function")


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
    # TODO: Person 4 implements
    raise NotImplementedError("Person 4: implement this function")


def compare_models(
    model_results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Generate comparison table/plot across all models.

    Args:
        model_results: {model_name: {"rmse": float, "mae": float, "param_count": int, ...}}
        save_path: If provided, save comparison to this path.
    """
    # TODO: Person 4 implements
    raise NotImplementedError("Person 4: implement this function")
