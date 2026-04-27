import csv

import numpy as np
import torch

from algo.common.sa import sample_ising
from algo.fmqa.fmqa import run_fmqa
from fmqa import PROJECT_ROOT
from objective import RandomHopfieldIsing


def main(
    num_spins,
    num_patterns,
    latent_dim,
    initial_dataset_size,
    seed: int,
    num_iter=400,
    train_epochs=2000,
    random_postprocess=True,
    tol=1e-6,
    stop_early=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    objective_function = RandomHopfieldIsing(
        num_spins=num_spins,
        num_patterns=num_patterns,
    )

    Jij = objective_function.J
    best_spins = sample_ising(Jij, num_reads=100)
    best_x = (best_spins + 1) / 2
    ground_state_energy = objective_function(best_x)
    print(f"Ground state energy: {ground_state_energy.item():.4f}")

    output_dir = (
        PROJECT_ROOT
        / "out"
        / "hopfield_ising"
        / (f"n_{num_spins}_p_{num_patterns}_k_{latent_dim}_init_{initial_dataset_size}")
        / str(seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # save the configuration of the objective function
    with (output_dir / "config_patterns.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "pattern", "value"])
        for i in range(objective_function.num_spins):
            for pattern in range(objective_function.num_patterns):
                writer.writerow(
                    [i, pattern, objective_function.patterns[i, pattern].item()]
                )

    with (output_dir / "config_Jij.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "Jij"])
        for i in range(objective_function.num_spins):
            for j in range(i + 1, objective_function.num_spins):
                writer.writerow([i, j, objective_function.J[i, j].item()])

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
        num_patterns=4,
        latent_dim=5,
        initial_dataset_size=5,
        seed=0,
        num_iter=400,
        train_epochs=2000,
        random_postprocess=True,
        stop_early=True,
    )
