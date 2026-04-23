import torch

def random_ising(x: torch.Tensor, J: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Calculate the energy of an Ising model configuration. The energy is given by
        E(x) = - sum_{i < j} J[i, j] * x[i] * x[j] - sum_i h[i] * x[i]

    Args:
        x: A tensor of shape (n,) or (d, n) representing the spin configuration(s). Each element should be either -1 or +1, where -1 represents a spin down and +1 represents a spin up.
        J: An upper-triangular tensor of shape (n, n) representing the interaction strengths between spins. J[i, j] is the coupling between spin i and spin j for i < j. The diagonal elements J[i, i] are ignored.
        h: A tensor of shape (n,) representing the external magnetic field on
            each spin.

    Returns:
        A scalar tensor representing the energy of the configuration x.
    """
    if torch.any((x != -1) & (x != 1)):
        raise ValueError("All elements of x must be either -1 or +1.")

    is_x_1d = x.ndim == 1
    if is_x_1d:
        x = x.unsqueeze(0)
    x = x.to(J.dtype)

    interaction_energy = - torch.einsum("ij,bi,bj->b", J, x, x)
    field_energy = - x @ h

    energy = interaction_energy + field_energy
    if is_x_1d:
        energy = energy.squeeze(0)
    return energy
