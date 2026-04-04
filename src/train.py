"""Shared training loop for all models."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import get_device


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device | None = None,
    save_path: str | None = None,
) -> dict:
    """Shared training function for all models.

    Args:
        model: Any model with interface (batch, window, 14) -> (batch, 1).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Max training epochs.
        lr: Initial learning rate.
        patience: Early stopping patience (epochs without val improvement).
        device: torch.device, auto-detected if None.
        save_path: Path to save best model checkpoint (.pt), skipped if None.

    Returns:
        Dict with keys: train_losses, val_losses, epoch_times, param_count,
        best_val_loss.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    train_losses: list[float] = []
    val_losses: list[float] = []
    epoch_times: list[float] = []
    best_val_loss = float("inf")
    es_counter = 0
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_running += loss.item() * x_batch.size(0)
        val_loss = val_running / len(val_loader.dataset)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}"
        )

        # --- Early stopping + checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            es_counter = 0
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            es_counter += 1
            if es_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    total_time = time.time() - total_start
    print(f"Best val loss: {best_val_loss:.6f} | Total time: {total_time:.1f}s")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "param_count": param_count,
        "best_val_loss": best_val_loss,
    }


def save_results(results: dict, path: str) -> None:
    """Save training results dict as JSON.

    Handles conversion of non-serializable types (e.g. numpy floats).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, list):
            serializable[k] = [float(x) for x in v]
        elif isinstance(v, (int, float, str)):
            serializable[k] = v
        else:
            serializable[k] = float(v)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
