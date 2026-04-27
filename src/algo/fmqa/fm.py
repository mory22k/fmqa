from __future__ import annotations

import torch
from torch import Tensor, nn


# Use Rendle's explicit gradients for FM to avoid autograd overhead and save memory.
class _FactorizationMachineFunction(torch.autograd.Function):
    """Factorization machine prediction with Rendle's explicit gradients."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        bias: Tensor,
        linear_weight: Tensor,
        linear_bias: Tensor | None,
        factors: Tensor,
    ) -> Tensor:
        linear_term = x @ linear_weight.squeeze(0) + bias.squeeze(0)
        if linear_bias is not None:
            linear_term = linear_term + linear_bias.squeeze(0)

        summed_factors = x @ factors
        squared_sum = x.pow(2) @ factors.pow(2)
        interaction_term = 0.5 * (summed_factors.pow(2) - squared_sum).sum(dim=1)

        ctx.save_for_backward(x, linear_weight, factors, summed_factors)
        ctx.has_linear_bias = linear_bias is not None
        return linear_term + interaction_term

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        x, linear_weight, factors, summed_factors = ctx.saved_tensors
        grad = grad_output.reshape(-1)

        # Rendle (2010): d y_hat / d v_{i,f}
        # = x_i * sum_j(v_{j,f} * x_j) - v_{i,f} * x_i^2.
        weighted_summed_factors = grad[:, None] * summed_factors
        grad_factors = x.T @ weighted_summed_factors
        grad_factors = grad_factors - (
            (grad[:, None] * x.pow(2)).sum(dim=0)[:, None] * factors
        )

        grad_bias = grad.sum().reshape(1)
        grad_linear_weight = (grad[:, None] * x).sum(dim=0, keepdim=True)
        grad_linear_bias = grad.sum().reshape(1) if ctx.has_linear_bias else None

        grad_x = grad[:, None] * (
            linear_weight + summed_factors @ factors.T - x * factors.pow(2).sum(dim=1)
        )

        return grad_x, grad_bias, grad_linear_weight, grad_linear_bias, grad_factors


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

        return _FactorizationMachineFunction.apply(
            x,
            self.bias,
            self.linear.weight,
            self.linear.bias,
            self.factors,
        )


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
        raise ValueError(
            "y must be a 1D tensor or a column tensor of shape (batch_size, 1)."
        )

    if y.shape[0] != features.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    targets = y.to(device=features.device, dtype=torch.float32)
    return features, targets


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def train_fm_model(
    model: FactorizationMachineRegressor,
    dataset_x: Tensor,
    dataset_y: Tensor,
    train_epochs: int = 2000,
    lr: float = 0.1,
    verbose: bool = False,
    tol_loss: float = 1e-12,
):
    if not torch.all((dataset_x == 0) | (dataset_x == 1)):
        raise ValueError("All elements of dataset_x must be either 0 or 1.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(train_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(dataset_x)
        loss = torch.nn.functional.mse_loss(predictions, dataset_y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch: 4d}, Loss: {loss.item():10.3e}")
        if loss.item() < tol_loss:
            break
