"""Training script for LSTM model — both configs."""

import sys

from src.dataset import prepare_data
from src.models.lstm import LSTMModel
from src.train import train_model, save_results
from src.utils import set_seed, get_device


def run_config(
    config_name: str,
    hidden_size: int,
    window_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    batch_size: int = 64,
) -> None:
    """Run a single LSTM training configuration."""
    print(f"\n{'='*60}")
    print(f"LSTM {config_name} | hidden={hidden_size} | window={window_size}")
    print(f"{'='*60}\n")

    set_seed(42)
    device = get_device()

    all_loaders, _ = prepare_data(batch_size=batch_size)
    train_loader, val_loader, _ = all_loaders[window_size]

    model = LSTMModel(
        input_size=14,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    save_path = f"results/lstm_{config_name}.pt"
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

    # Add config metadata
    results["config"] = {
        "name": config_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "window_size": window_size,
        "batch_size": batch_size,
    }

    save_results(results, f"results/lstm_{config_name}.json")
    print(f"\nResults saved to results/lstm_{config_name}.json")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "all"

    if config in ("A", "all"):
        run_config("configA", hidden_size=64)

    if config in ("B", "all"):
        run_config("configB", hidden_size=128)
