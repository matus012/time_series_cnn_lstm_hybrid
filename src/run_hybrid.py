"""Training script for hybrid CNN-LSTM model configurations.

This script trains two hybrid variants for multivariate time-series forecasting
on the Jena Climate dataset.

Hybrid-specific design choices:
    - window_size = 48
    - cnn_channels = 64
    - kernel_size = 5
    - lstm_hidden_size = 32
    - MaxPool1d with kernel size 2 inside the model

The two tested configurations differ only in the number of LSTM layers:
    - configA: 2 LSTM layers
    - configB: 3 LSTM layers

This allows a clean comparison of LSTM depth inside the hybrid setting.
"""

import sys

from src.dataset import prepare_data
from src.models.hybrid import HybridModel
from src.train import train_model, save_results
from src.utils import get_device, set_seed


def run_config(
    config_name: str,
    lstm_num_layers: int,
    window_size: int = 48,
    cnn_channels: int = 64,
    kernel_size: int = 5,
    lstm_hidden_size: int = 32,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    batch_size: int = 64,
) -> None:
    """Run a single hybrid CNN-LSTM training configuration."""
    print(f"\n{'=' * 60}")
    print(f"Hybrid CNN-LSTM {config_name} | num_layers={lstm_num_layers} | window={window_size}")
    print(f"{'=' * 60}\n")

    set_seed(42)
    device = get_device()

    all_loaders, _ = prepare_data(batch_size=batch_size)
    train_loader, val_loader, _ = all_loaders[window_size]

    model = HybridModel(
        input_size=14,
        cnn_channels=cnn_channels,
        kernel_size=kernel_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dropout=dropout,
    )

    save_path = f"results/hybrid_{config_name}.pt"
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        device=device,
        save_path=save_path,
    )

    results["config"] = {
        "name": config_name,
        "cnn_channels": cnn_channels,
        "kernel_size": kernel_size,
        "lstm_hidden_size": lstm_hidden_size,
        "lstm_num_layers": lstm_num_layers,
        "dropout": dropout,
        "lr": lr,
        "window_size": window_size,
        "batch_size": batch_size,
    }

    save_results(results, f"results/hybrid_{config_name}.json")
    print(f"\nResults saved to results/hybrid_{config_name}.json")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "all"

    if config in ("A", "all"):
        run_config("configA", lstm_num_layers=2)

    if config in ("B", "all"):
        run_config("configB", lstm_num_layers=3)