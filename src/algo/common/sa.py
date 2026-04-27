import torch
from dwave.samplers import SimulatedAnnealingSampler


def sample_ising(
    J: torch.Tensor, h: torch.Tensor | None = None, return_multiple=False, **kwargs
) -> torch.Tensor:
    """
    Sample from an Ising model using D-Wave's Simulated Annealing Sampler.
    The Ising model is defined by the energy function:
        E(x) = - sum_{i < j} J[i, j] * x[i] * x[j] - sum_i h[i] * x[i]
    The D-Wave sampler expects the energy function in the form:
        E(x) = sum_{i < j} J[i, j] * x[i] * x[j] + sum_i h[i] * x[i]
    Therefore, we need to negate J and h when passing them to the sampler.

    Args:
        J: A 2D tensor representing the coupling strengths between spins. Should be upper-triangular.
        h: A 1D tensor representing the external magnetic field on each spin.
        **kwargs: Additional keyword arguments to pass to the sampler (e.g., num_reads).

    Returns:
        A tensor containing the sampled spin configuration(s). Each element is either -1 or +1.
    """
    if h is None:
        h = torch.zeros(J.shape[0], device=J.device)
    sa_sampler = SimulatedAnnealingSampler()
    response = sa_sampler.sample_ising(-h.cpu().numpy(), -J.cpu().numpy(), **kwargs)
    record = response.record
    record.sort(order="energy")
    best_sample = record[0].sample
    if return_multiple:
        return torch.stack(
            [torch.from_numpy(sample).to(J.device) for sample in record.sample]
        )
    return torch.from_numpy(best_sample).to(J.device)


def sample_qubo(Q, return_multiple=False, **kwargs):
    """
    Sample from a QUBO model using D-Wave's Simulated Annealing Sampler.
    The QUBO model is defined by the energy function:
        E(x) = sum_{i <= j} Q[i, j] * x[i] * x[j]
    The D-Wave sampler expects the energy function in the form:
        E(x) = sum_{i <= j} Q[i, j] * x[i] * x[j]
    Therefore, we can directly pass Q to the sampler.

    Args:
        Q: A 2D tensor representing the QUBO coefficients. Should be upper-triangular.
        **kwargs: Additional keyword arguments to pass to the sampler (e.g., num_reads).

    Returns:
        A tensor containing the sampled binary configuration(s). Each element is either 0 or 1.
    """

    sa_sampler = SimulatedAnnealingSampler()
    response = sa_sampler.sample_qubo(Q.cpu().numpy(), **kwargs)
    record = response.record
    record.sort(order="energy")
    best_sample = record[0].sample
    if return_multiple:
        return torch.stack(
            [torch.from_numpy(sample).to(Q.device) for sample in record.sample]
        )
    return torch.from_numpy(best_sample).to(Q.device)
