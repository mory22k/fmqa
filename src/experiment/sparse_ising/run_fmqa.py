import csv

import numpy as np
import torch

from algo.common.sa import sample_ising
from algo.fmqa.fmqa import run_fmqa
from fmqa import PROJECT_ROOT
from objective import RandomSparseIsing


def main(
    num_spins,
    latent_dim,
    initial_dataset_size,
    seed: int,
    num_iter=400,
    train_epochs=2000,
    random_postprocess=True,
    tol=1e-6,
    stop_early=True,
    density=1.0,
    include_local_field=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # generate the objective function and find the ground state energy
    objective_function = RandomSparseIsing(
        num_spins=num_spins,
        density=density,
        include_local_field=include_local_field,
    )
    Jij, hi = objective_function.J, objective_function.h
    best_spins = sample_ising(Jij, hi, num_reads=100)
    best_x = (best_spins + 1) / 2
    ground_state_energy = objective_function(best_x)
    print(f"Ground state energy: {ground_state_energy.item():.4f}")

    output_dir = (
        PROJECT_ROOT
        / "out"
        / "sparse_ising"
        / (
            f"n_{num_spins}_k_{latent_dim}_init_{initial_dataset_size}_"
            f"d_{density}_h_{int(include_local_field)}"
        )
        / str(seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # save the configuration of the objective function
    with (output_dir / "config_Jij.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "Jij"])
        for i in range(objective_function.num_spins):
            for j in range(i + 1, objective_function.num_spins):
                writer.writerow([i, j, objective_function.J[i, j].item()])

    with (output_dir / "config_hi.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "hi"])
        for i in range(objective_function.num_spins):
            writer.writerow([i, objective_function.h[i].item()])

    run_fmqa(
        objective_function,
        input_dim=num_spins,
        latent_dim=latent_dim,
        initial_dataset_size=initial_dataset_size,
        num_iter=num_iter,
        train_epochs=train_epochs,
        output_dir=output_dir,
        ground_truth=ground_state_energy,
        stop_early=stop_early,
        random_postprocess=random_postprocess,
        tol=tol,
    )


if __name__ == "__main__":
    main(
        num_spins=32,
        latent_dim=16,
        initial_dataset_size=5,
        seed=0,
        num_iter=400,
        train_epochs=2000,
        density=0.1,
        include_local_field=False,
        random_postprocess=True,
        stop_early=True,
    )
