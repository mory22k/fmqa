from __future__ import annotations

import itertools

import torch

from algo.fmqa.fm import FactorizationMachineRegressor
from algo.fmqa.fmqa import fm_to_qubo


def test_fm_to_qubo_matches_fm_plus_bias() -> None:
    model = FactorizationMachineRegressor(input_dim=3, latent_dim=2)

    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.25, -0.5, 2.0]]))
        model.linear.bias.copy_(torch.tensor([0.75]))
        model.bias.copy_(torch.tensor([1.5]))
        model.factors.copy_(
            torch.tensor(
                [
                    [1.0, 0.5],
                    [0.25, -1.0],
                    [1.5, 0.75],
                ]
            )
        )

    qubo = fm_to_qubo(
        linear_weight=model.linear.weight.squeeze(0),
        factor=model.factors,
    )
    bias = (model.linear.bias + model.bias).squeeze()

    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=3)))
    fm_values = model(x)
    qubo_values = bias + (torch.triu(x[:, :, None] * x[:, None, :]) * qubo).sum(
        dim=(1, 2)
    )

    print("FM values:", fm_values)
    print("QUBO values:", qubo_values)

    assert torch.allclose(fm_values, qubo_values)


if __name__ == "__main__":
    test_fm_to_qubo_matches_fm_plus_bias()
