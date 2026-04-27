from __future__ import annotations

import torch
import torch.nn.functional as F

from algo.fmqa.fm import FactorizationMachineRegressor


def _reference_fm(
    x: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    factors: torch.Tensor,
) -> torch.Tensor:
    linear_term = (x @ linear_weight.squeeze(0)) + linear_bias.squeeze(0) + bias
    summed_factors = x @ factors
    squared_sum = x.pow(2) @ factors.pow(2)
    interaction_term = 0.5 * (summed_factors.pow(2) - squared_sum).sum(dim=1)
    return linear_term + interaction_term


def test_factorization_machine_backward_matches_autograd_reference() -> None:
    torch.manual_seed(0)

    model = FactorizationMachineRegressor(input_dim=4, latent_dim=3).double()
    x = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    target = torch.randn(5, dtype=torch.double)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_bias = model.bias.detach().clone().requires_grad_(True)
    ref_linear_weight = model.linear.weight.detach().clone().requires_grad_(True)
    ref_linear_bias = model.linear.bias.detach().clone().requires_grad_(True)
    ref_factors = model.factors.detach().clone().requires_grad_(True)

    loss = F.mse_loss(model(x), target)
    loss.backward()

    ref_loss = F.mse_loss(
        _reference_fm(
            ref_x,
            ref_bias,
            ref_linear_weight,
            ref_linear_bias,
            ref_factors,
        ),
        target,
    )
    ref_loss.backward()

    assert torch.allclose(x.grad, ref_x.grad)
    assert torch.allclose(model.bias.grad, ref_bias.grad)
    assert torch.allclose(model.linear.weight.grad, ref_linear_weight.grad)
    assert torch.allclose(model.linear.bias.grad, ref_linear_bias.grad)
    assert torch.allclose(model.factors.grad, ref_factors.grad)
