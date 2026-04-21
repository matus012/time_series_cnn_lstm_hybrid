"""CNN-LSTM hybrid model for time series prediction. (Person 3)"""

import torch
import torch.nn as nn


class HybridModel(nn.Module):
    """Hybrid CNN-LSTM model for univariate time series forecasting.

    Input:  (batch, window_size, input_size) — all 14 features
    Output: (batch, 1) — predicted temperature (scaled)

    Architecture:
        - CNN block: two stacked Conv1D layers with ReLU, followed by MaxPool1d.
          Extracts local temporal patterns across the input window.
        - LSTM block: multi-layer LSTM with dropout between layers.
          Models long-range dependencies on CNN-extracted features.
        - Takes hidden state from last timestep.
        - Single linear layer maps hidden -> 1 output.

    Rationale:
        The CNN captures short-horizon local patterns. The LSTM then models
        long-horizon dependencies on these higher-level features.
        Parameters are hybrid-specific (not copied from standalone LSTM/CNN):
          - Larger input window (48) compensates for MaxPool sequence halving,
            keeping 24 timesteps for the LSTM — same as standalone LSTM's view.
          - Smaller LSTM hidden size (32) since CNN already compressed features.
          - Wider kernel (5) gives richer per-feature context before LSTM.

    Shape transformations inside forward():
        (batch, 48, 14)          — input (window_size=48)
        (batch, 14, 48)          — permuted for Conv1D
        (batch, cnn_ch, 24)      — after CNN + MaxPool
        (batch, 24, cnn_ch)      — permuted back for LSTM
        (batch, lstm_hidden)     — last LSTM hidden state
        (batch, 1)               — final prediction
    """

    def __init__(
        self,
        input_size: int = 14,
        cnn_channels: int = 64,
        kernel_size: int = 5,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            input_size: Number of input features per time step (14 for Jena).
            cnn_channels: Output channels for both Conv1D layers.
            kernel_size: Conv1D kernel size (odd; padding keeps sequence length).
            lstm_hidden_size: Hidden size of LSTM.
            lstm_num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
        """
        super().__init__()

        padding = kernel_size // 2  # "same" padding: output length == input length

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # halves sequence length (48 -> 24)
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        # Conv1D expects (batch, channels, length)
        x = x.permute(0, 2, 1)              # (batch, input_size, window)

        # CNN feature extraction
        x = self.cnn(x)                     # (batch, cnn_channels, window/2)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)              # (batch, window/2, cnn_channels)

        # Sequential modeling
        out, _ = self.lstm(x)               # (batch, window/2, lstm_hidden)
        out = out[:, -1, :]                 # last timestep: (batch, lstm_hidden)
        out = self.fc(out)                  # (batch, 1)
        return out
