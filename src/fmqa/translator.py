from __future__ import annotations

import torch
from torch import Tensor


def fm_to_qubo(linear_weight: Tensor, factor: Tensor) -> Tensor:
    """Convert trained FM parameters into an upper-triangular QUBO matrix.

    The returned matrix ``Q`` follows the convention
    ``sum_{i <= j} Q[i, j] * x_i * x_j`` for binary variables.

    The FM bias term is not included because it is a constant offset and does
    not change the optimizer's argmin.
    """

    linear = torch.as_tensor(linear_weight)
    factors = torch.as_tensor(factor)

    if factors.ndim != 2:
        raise ValueError(
            "factor must be a 2D tensor of shape (input_dim, latent_dim)."
        )

    linear = linear.reshape(-1).to(device=factors.device, dtype=factors.dtype)
    if linear.shape[0] != factors.shape[0]:
        raise ValueError(
            "linear_weight and factor must agree on input_dim: "
            f"got {linear.shape[0]} and {factors.shape[0]}."
        )

    # For binary FM inputs, the coefficient of x_i x_j (i < j) is <v_i, v_j>.
    interaction = factors @ factors.T
    qubo = torch.triu(interaction, diagonal=1)
    qubo = qubo + torch.diag(linear)
    return qubo


__all__ = ["fm_to_qubo"]
