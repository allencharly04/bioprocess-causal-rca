"""
LSTM autoencoder for multivariate time-series anomaly detection.

Architecture: sequence-to-sequence with full encoder output passed to decoder
(rather than compressing to a single hidden vector). This is the standard
TEP-paper architecture and trains far more easily than the bottleneck variant.

Trained only on normal-operation windows. At inference, reconstruction error
on a window is the deviation score: higher error = more anomalous.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int = 52,
        hidden_dim: int = 64,
        n_layers: int = 2,
        window_size: int = 50,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # Encoder: returns full sequence of hidden states
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Decoder: same shape, takes encoder's full output sequence as input
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Project decoder hidden states back to feature space
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window_size, n_features)
        returns: (batch, window_size, n_features) reconstruction
        """
        # Encode: full sequence of hidden states
        encoded, _ = self.encoder(x)            # (B, T, H)
        decoded, _ = self.decoder(encoded)      # (B, T, H)
        reconstruction = self.output_proj(decoded)  # (B, T, F)
        return reconstruction

    def reconstruction_error(
        self, x: torch.Tensor, reduction: str = "per_window"
    ) -> torch.Tensor:
        """
        reduction:
          "per_window"   -> (batch,)        mean MSE per window  [scoring]
          "per_timestep" -> (batch, T)      mean MSE per timestep
          "per_feature"  -> (batch, F)      mean MSE per feature [for RCA]
          "full"         -> (batch, T, F)   raw squared error
        """
        x_hat = self.forward(x)
        sq_err = (x - x_hat) ** 2
        if reduction == "per_window":
            return sq_err.mean(dim=(1, 2))
        elif reduction == "per_timestep":
            return sq_err.mean(dim=2)
        elif reduction == "per_feature":
            return sq_err.mean(dim=1)
        elif reduction == "full":
            return sq_err
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSTMAutoencoder()
    x = torch.randn(4, 50, 52)
    y = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters:   {count_params(model):,}")