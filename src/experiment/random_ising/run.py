from fmqa.fm import FactorizationMachineRegressor
from fmqa.translator import fm_to_qubo
from fmqa.annealing import sa_ising, sa_qubo
from obj_func import random_ising
import torch
from tqdm import tqdm

def prepare_random_ising_model(N, device=None):
    J = torch.triu(torch.randn(N, N, device=device), diagonal=1)
    h = torch.zeros(N, device=device)
    return J, h

def train_fm_model(model, spins, energies, train_epochs=4000, lr=0.1, verbose=False, tol_loss=1e-12):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(train_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(spins)
        loss = torch.nn.functional.mse_loss(predictions, energies)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch: 4d}, Loss: {loss.item():10.3e}")
        if loss.item() < tol_loss:
            break

def main(num_spins, latent_dim, initial_dataset_size, num_iter = 200, random_postprocess=True, tol=1e-6):
    J, h = prepare_random_ising_model(N=num_spins)
    best_sample = sa_ising(J, h, num_reads=100)
    ground_truth = random_ising(best_sample, J, h)

    # initialize the dataset with random samples
    spins = torch.randint(0, 2, (initial_dataset_size, num_spins)).float() * 2 - 1
    energies = random_ising(spins, J, h)

    # prepare the progress bar
    progress_bar = tqdm(range(num_iter), desc="FMQA Iteration", disable=False)

    # main optimization loop
    for iteration in progress_bar:
        model = FactorizationMachineRegressor(input_dim=num_spins, latent_dim=latent_dim)
        # train the FM model and translate to QUBO
        train_fm_model(model, spins, energies, verbose=False)
        linear_weight = model.linear.weight.squeeze(0)
        factor = model.factors
        qubo = fm_to_qubo(linear_weight.detach().cpu(), factor.detach().cpu())

        # sample from the QUBO
        next_vars = sa_qubo(qubo, num_reads=100)
        next_spins = next_vars.float() * 2 - 1  # convert {0,1} to {-1,1}

        # propose a random sample if the next sample is already in the dataset
        if random_postprocess and torch.any(torch.all(spins == next_spins, dim=1)):
            progress_bar.write(f"Iter {iteration + 1:4d} | Sample already in dataset, proposing a random sample instead.")
            next_spins = torch.randint(0, 2, (1, num_spins)).float().squeeze(0) * 2 - 1

        # evaluate the next sample and update the dataset
        next_energy = random_ising(next_spins, J, h)
        spins = torch.cat([spins, next_spins.unsqueeze(0)], dim=0)
        energies = torch.cat([energies, next_energy.unsqueeze(0)], dim=0)

        progress_bar.write(f"Iter {iteration + 1:4d} | Next: {next_energy.item():10.3e}, Best: {torch.min(energies).item():10.3e}, Ground: {ground_truth.item():10.3e}, Regret: {torch.min(energies).item() - ground_truth.item():10.3e}")

        if torch.min(energies) - ground_truth < tol:
            progress_bar.write(f"Converged to the ground truth within tolerance {tol:.1e}. Stopping optimization.")
            break

if __name__ == "__main__":
    main(num_spins=20, latent_dim=5, initial_dataset_size=20, num_iter=400, random_postprocess=True)
