from __future__ import annotations

import torch
from torch import Tensor, nn


class FactorizationMachineRegressor(nn.Module):
    """Naive factorization machine regressor for dense tabular inputs."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(input_dim, 1)
        self.factors = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor of shape (batch_size, input_dim).")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"x.shape[1] must be {self.input_dim}, but got {x.shape[1]}."
            )

        linear_term = self.linear(x).squeeze(-1) + self.bias

        summed_factors = x @ self.factors
        summed_factors_sq = summed_factors.pow(2)
        squared_inputs = x.pow(2)
        squared_factors = self.factors.pow(2)
        squared_sum = squared_inputs @ squared_factors
        interaction_term = 0.5 * (summed_factors_sq - squared_sum).sum(dim=1)

        return linear_term + interaction_term

    @torch.no_grad()
    def predict(self, x: Tensor, *, device: torch.device | str | None = None) -> Tensor:
        features = _prepare_features(x=x, input_dim=self.input_dim, device=device)
        self.eval()
        predictions = self(features)
        return predictions.cpu()


def _prepare_features(
    *,
    x: Tensor,
    input_dim: int,
    device: torch.device | str | None,
) -> Tensor:
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor of shape (batch_size, input_dim).")
    if x.shape[1] != input_dim:
        raise ValueError(f"x.shape[1] must be {input_dim}, but got {x.shape[1]}.")

    target_device = _resolve_device(device)
    return x.to(device=target_device, dtype=torch.float32)


def _prepare_training_tensors(
    *,
    x: Tensor,
    y: Tensor,
    input_dim: int,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor]:
    features = _prepare_features(x=x, input_dim=input_dim, device=device)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim != 1:
        raise ValueError("y must be a 1D tensor or a column tensor of shape (batch_size, 1).")

    if y.shape[0] != features.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    targets = y.to(device=features.device, dtype=torch.float32)
    return features, targets


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)
