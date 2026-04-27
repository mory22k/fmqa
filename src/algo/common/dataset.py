from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def prepare_initial_dataset(
    objective_function: Callable[[Tensor], Tensor],
    *,
    input_dim: int,
    dataset_size: int,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """Prepare an initial binary dataset by randomly sampling and evaluating."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative.")

    target_device = torch.device("cpu") if device is None else torch.device(device)
    dataset_x = torch.empty(
        (dataset_size, input_dim), device=target_device, dtype=torch.float32
    )
    dataset_y_parts: list[Tensor] = []

    for index in range(dataset_size):
        x = torch.randint(
            0,
            2,
            (input_dim,),
            device=target_device,
            dtype=torch.float32,
        )
        y = objective_function(x)
        y = y.to(device=target_device, dtype=torch.float32).reshape(())

        dataset_x[index] = x
        dataset_y_parts.append(y)

        if verbose:
            print(f"Sample {index + 1:4d}/{dataset_size:4d} | {y.item():10.3e}")

    if not dataset_y_parts:
        return dataset_x, torch.empty((0,), device=target_device, dtype=torch.float32)

    return dataset_x, torch.stack(dataset_y_parts)
