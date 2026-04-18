"""Training script for CNN model — both configs."""

import sys

from src.dataset import prepare_data
from src.models.cnn import CNNModel
from src.train import train_model, save_results
from src.utils import set_seed, get_device


def run_config(
    config_name: str,
    channels_1: int,
    channels_2: int,
    kernel_size_1: int,
    kernel_size_2: int,
    window_size: int = 24,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    batch_size: int = 64,
) -> None:
    """Run a single CNN training configuration."""
    print(f"\n{'=' * 60}")
    print(
        f"CNN {config_name} | c1={channels_1} | c2={channels_2} | "
        f"k1={kernel_size_1} | k2={kernel_size_2} | window={window_size}"
    )
    print(f"{'=' * 60}\n")

    set_seed(42)
    device = get_device()

    all_loaders, target_col = prepare_data(batch_size=batch_size)
    train_loader, val_loader, test_loader = all_loaders[window_size]

    model = CNNModel(
        input_size=14,
        channels_1=channels_1,
        channels_2=channels_2,
        kernel_size_1=kernel_size_1,
        kernel_size_2=kernel_size_2,
        dropout=dropout,
    )

    save_path = f"results/cnn_{config_name}.pt"
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
        "channels_1": channels_1,
        "channels_2": channels_2,
        "kernel_size_1": kernel_size_1,
        "kernel_size_2": kernel_size_2,
        "dropout": dropout,
        "lr": lr,
        "window_size": window_size,
        "batch_size": batch_size,
    }

    save_results(results, f"results/cnn_{config_name}.json")
    print(f"\nResults saved to results/cnn_{config_name}.json")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "all"
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    if config in ("A", "all"):
        run_config(
            config_name="configA",
            channels_1=32,
            channels_2=64,
            kernel_size_1=3,
            kernel_size_2=3,
            dropout=0.2,
            window_size=window_size,
        )

    if config in ("B", "all"):
        run_config(
            config_name="configB",
            channels_1=64,
            channels_2=128,
            kernel_size_1=5,
            kernel_size_2=3,
            dropout=0.3,
            window_size=window_size,
        )