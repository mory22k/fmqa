from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from algo.common import prepare_initial_dataset
from algo.common.sa import sample_qubo
from algo.fmqa.fm import FactorizationMachineRegressor, train_fm_model


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
        raise ValueError("factor must be a 2D tensor of shape (input_dim, latent_dim).")

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


def run_fmqa(
    objective_function: Callable[[Tensor], Tensor],
    *,
    input_dim: int,
    latent_dim: int,
    initial_dataset_size: int,
    num_iter: int,
    train_epochs: int,
    output_dir: Path,
    ground_truth: Tensor | None = None,
    stop_early: bool = False,
    random_postprocess: bool = True,
    tol: float = 1e-6,
) -> None:
    if ground_truth is not None:
        ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32)

    write_progress_header(output_dir)

    print("Preparing initial dataset...")
    dataset_x, dataset_y = prepare_initial_dataset(
        objective_function,
        input_dim=input_dim,
        dataset_size=initial_dataset_size,
        verbose=True,
    )

    progress_bar = tqdm(range(num_iter), desc="FMQA Iteration", disable=False)
    print("Starting optimization...")
    for iteration in progress_bar:
        model = FactorizationMachineRegressor(
            input_dim=input_dim, latent_dim=latent_dim
        )
        train_fm_model(
            model, dataset_x, dataset_y, train_epochs=train_epochs, verbose=False
        )
        qubo = fm_to_qubo(
            model.linear.weight.squeeze(0).detach().cpu(),
            model.factors.detach().cpu(),
        )

        next_x = sample_qubo(qubo, num_reads=10)
        if random_postprocess and torch.any(torch.all(dataset_x == next_x, dim=1)):
            progress_bar.write(
                f"Iter {iteration + 1:4d} | Sample already in dataset, proposing a random sample instead."
            )
            next_x = torch.randint(0, 2, (1, input_dim)).float().squeeze(0)

        next_y = objective_function(next_x)
        append_progress_csv(output_dir, iteration + 1, next_y.item())

        dataset_x = torch.cat([dataset_x, next_x.unsqueeze(0)], dim=0)
        dataset_y = torch.cat([dataset_y, next_y.unsqueeze(0)], dim=0)

        message = (
            f"Iter {iteration + 1:4d} | "
            f"Next: {next_y.item():10.3e}, "
            f"Best: {torch.min(dataset_y).item():10.3e}"
        )
        if ground_truth is not None:
            regret = torch.min(dataset_y) - ground_truth
            message += (
                f", Ground: {ground_truth.item():10.3e}, Regret: {regret.item():10.3e}"
            )
        progress_bar.write(message)

        if stop_early and ground_truth is not None and regret < tol:
            progress_bar.write(
                f"Regret converged to {regret.item():.1e} < {tol:.1e}. Stopping optimization."
            )
            break


def write_progress_header(output_dir: Path) -> None:
    with (output_dir / "progress.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["iteration", "next_y"])


def append_progress_csv(output_dir: Path, iteration: int, next_y: float) -> None:
    with (output_dir / "progress.csv").open("a", newline="") as f:
        csv.writer(f).writerow([iteration, next_y])
