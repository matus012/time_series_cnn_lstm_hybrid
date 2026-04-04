"""Data pipeline: loading, preprocessing, splitting, sliding window dataset."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

TARGET_COL_NAME: str = "T (degC)"


def load_and_preprocess(csv_path: str, subsample: int = 6) -> pd.DataFrame:
    """Load CSV, parse datetime index, subsample, keep only numeric columns.

    Args:
        csv_path: Path to the raw Jena climate CSV.
        subsample: Keep every n-th row (default 6 = hourly from 10-min data).

    Returns:
        DataFrame with DatetimeIndex and 14 numeric feature columns.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date Time"], index_col="Date Time")
    print(f"Raw shape: {df.shape}")

    df = df.iloc[::subsample]
    print(f"After subsampling (every {subsample}th): {df.shape}")

    df = df.select_dtypes(include=[np.number])
    df = df.ffill()
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split — no shuffling.

    Args:
        df: Full preprocessed DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (rest goes to test).

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, save_dir: str = "results") -> StandardScaler:
    """Fit StandardScaler on training data only, persist scaler and target stats.

    Args:
        train_df: Training split DataFrame.
        save_dir: Directory to save scaler.pkl and target_stats.pkl.

    Returns:
        Fitted StandardScaler.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(train_df.values)

    target_idx = train_df.columns.get_loc(TARGET_COL_NAME)
    target_stats = {
        "mean": float(scaler.mean_[target_idx]),
        "std": float(scaler.scale_[target_idx]),
    }

    joblib.dump(scaler, save_path / "scaler.pkl")
    joblib.dump(target_stats, save_path / "target_stats.pkl")
    print(f"Scaler saved to {save_path / 'scaler.pkl'}")
    print(f"Target stats ({TARGET_COL_NAME}): {target_stats}")
    return scaler


def transform_data(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform DataFrame using a fitted scaler.

    Args:
        df: DataFrame to transform.
        scaler: Previously fitted StandardScaler.

    Returns:
        Scaled numpy array of shape (N, num_features).
    """
    return scaler.transform(df.values)


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for multivariate time series.

    Each sample: x = window of shape (window_size, num_features),
                 y = next-step target scalar of shape (1,).
    """

    def __init__(self, data: np.ndarray, window_size: int, target_col: int) -> None:
        """
        Args:
            data: Scaled array of shape (N, num_features).
            window_size: Number of time steps in input window.
            target_col: Column index of the prediction target.
        """
        self.data = data
        self.window_size = window_size
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size, self.target_col]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap datasets into DataLoaders.

    Args:
        train_ds: Training dataset (shuffled).
        val_ds: Validation dataset.
        test_ds: Test dataset.
        batch_size: Batch size for all loaders.

    Returns:
        (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_all_dataloaders(
    train_scaled: np.ndarray,
    val_scaled: np.ndarray,
    test_scaled: np.ndarray,
    target_col: int,
    batch_size: int = 64,
) -> dict[int, tuple[DataLoader, DataLoader, DataLoader]]:
    """Build DataLoaders for both window sizes (24 and 48) — ablation handoff.

    Args:
        train_scaled: Scaled training array.
        val_scaled: Scaled validation array.
        test_scaled: Scaled test array.
        target_col: Target column index.
        batch_size: Batch size.

    Returns:
        Dict mapping window_size -> (train_loader, val_loader, test_loader).
    """
    loaders: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}
    for ws in (24, 48):
        train_ds = TimeSeriesDataset(train_scaled, ws, target_col)
        val_ds = TimeSeriesDataset(val_scaled, ws, target_col)
        test_ds = TimeSeriesDataset(test_scaled, ws, target_col)
        loaders[ws] = get_dataloaders(train_ds, val_ds, test_ds, batch_size)
        print(f"Window {ws}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} samples")
    return loaders


def prepare_data(
    csv_path: str = "data/jena_climate_2009_2016.csv",
    subsample: int = 6,
    batch_size: int = 64,
    save_dir: str = "results",
) -> tuple[dict[int, tuple[DataLoader, DataLoader, DataLoader]], int]:
    """End-to-end data pipeline: load -> split -> scale -> dataloaders.

    Single entry point for teammates. Chains all pipeline steps.

    Args:
        csv_path: Path to raw CSV.
        subsample: Subsampling factor.
        batch_size: Batch size for DataLoaders.
        save_dir: Directory for persisted artifacts.

    Returns:
        (all_loaders_dict, target_col_index) where all_loaders_dict maps
        window_size -> (train_loader, val_loader, test_loader).
    """
    df = load_and_preprocess(csv_path, subsample)
    train_df, val_df, test_df = split_data(df)
    scaler = fit_scaler(train_df, save_dir)

    target_col = train_df.columns.get_loc(TARGET_COL_NAME)

    train_scaled = transform_data(train_df, scaler)
    val_scaled = transform_data(val_df, scaler)
    test_scaled = transform_data(test_df, scaler)

    all_loaders = get_all_dataloaders(train_scaled, val_scaled, test_scaled, target_col, batch_size)
    return all_loaders, target_col
