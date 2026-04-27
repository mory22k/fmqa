from __future__ import annotations

import torch

from algo.common.sa import sample_ising


class RandomSparseIsing:
    """Sparse random Ising objective for binary FMQA inputs."""

    def __init__(
        self,
        num_spins: int,
        *,
        density: float = 1.0,
        include_local_field: bool = False,
        ground_state_num_reads: int = 100,
    ) -> None:
        if num_spins <= 0:
            raise ValueError("num_spins must be positive.")
        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be between 0.0 and 1.0.")

        self.num_spins = num_spins
        self.ground_state_num_reads = ground_state_num_reads

        mask = torch.rand(num_spins, num_spins) < density
        J = torch.randn(num_spins, num_spins) * mask
        self.J = torch.triu(J.float(), diagonal=1)

        h = torch.randn(num_spins) if include_local_field else torch.zeros(num_spins)
        self.h = h.float()
        self.ground_state_spins: torch.Tensor | None = None
        self._ground_state_energy: torch.Tensor | None = None

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        is_1d = spins.ndim == 1
        if is_1d:
            spins = spins.unsqueeze(0)

        spins = spins.to(dtype=self.J.dtype)
        interaction = -torch.einsum("ij,bi,bj->b", self.J, spins, spins)
        field = -(spins @ self.h)
        energy = interaction + field
        return energy.squeeze(0) if is_1d else energy

    def sample_ground_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        spins = sample_ising(self.J, self.h, num_reads=self.ground_state_num_reads)
        energy = self.energy(spins)
        self.ground_state_spins = spins
        self._ground_state_energy = energy
        return spins, energy

    @property
    def ground_state_energy(self) -> torch.Tensor:
        if self._ground_state_energy is None:
            raise RuntimeError(
                "Call sample_ground_state() before accessing ground_state_energy."
            )
        return self._ground_state_energy

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        spins = x.float() * 2 - 1
        return self.energy(spins)
