"""LSTM model for time series prediction."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM model for multivariate time series forecasting.

    Input:  (batch, window_size, input_size) — all 14 features
    Output: (batch, 1) — predicted temperature

    Architecture:
        - Multi-layer LSTM with dropout between layers
        - Takes hidden state from last timestep
        - Single linear layer maps hidden -> 1 output
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        out, _ = self.lstm(x)       # (batch, window, hidden)
        out = out[:, -1, :]         # last timestep: (batch, hidden)
        out = self.fc(out)          # (batch, 1)
        return out
