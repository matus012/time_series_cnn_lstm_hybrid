"""CNN model for time series prediction. (Person 2)"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """1D-CNN model for multivariate time series forecasting.

    Input:  (batch, window_size, input_size) — all 14 features
    Output: (batch, 1) — predicted temperature

    Architecture:
        - Permutes input from (batch, window, features) to (batch, features, window)
        - Two Conv1d layers with ReLU activations
        - Adaptive average pooling over time dimension
        - Dropout + linear layer for final regression output
    """

    def __init__(
        self,
        input_size: int = 14,
        channels_1: int = 32,
        channels_2: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=channels_1,
            kernel_size=kernel_size_1,
            padding=kernel_size_1 // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels_1,
            out_channels=channels_2,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        x = x.permute(0, 2, 1)      # (batch, features, window)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, channels_2)
        x = self.dropout(x)
        x = self.fc(x)              # (batch, 1)
        return x