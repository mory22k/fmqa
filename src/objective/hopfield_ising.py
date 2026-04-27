from __future__ import annotations

import torch

from objective.base import ObjectiveFunction


class RandomHopfieldIsing(ObjectiveFunction):
    """Random Hopfield Ising objective for binary FMQA inputs."""

    def __init__(
        self,
        num_spins: int,
        num_patterns: int,
    ) -> None:
        if num_spins <= 0:
            raise ValueError("num_spins must be positive.")
        if num_patterns <= 0:
            raise ValueError("num_patterns must be positive.")

        self.num_spins = num_spins
        self.num_patterns = num_patterns

        self.patterns = torch.randint(0, 2, (num_spins, num_patterns)).float() * 2 - 1
        self.J = torch.triu(self.patterns @ self.patterns.T / num_spins, diagonal=1)

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        is_1d = spins.ndim == 1
        if is_1d:
            spins = spins.unsqueeze(0)

        spins = spins.float()
        energy = -torch.einsum("ij,bi,bj->b", self.J, spins, spins)
        return energy.squeeze(0) if is_1d else energy

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        spins = x.float() * 2 - 1
        return self.energy(spins)
