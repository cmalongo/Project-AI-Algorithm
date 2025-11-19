"""
AI Algorithms – Option A (Model Compression: Pruning)
-----------------------------------------------------

This script implements a *toy* example of:
- A small MLP classifier on MNIST
- Magnitude-based pruning of weights
- A Genetic Algorithm (GA) that searches for a good pruning rate

It is designed to satisfy the requirements of the
"AI Algorithms Project Statement 2025" – Option A:
- We choose the facet "Model Compression (pruning)"
- We implement one AI decision-making technique (Genetic Algorithm)
  to optimize a specific sub-problem: choosing a pruning rate.

You can:
- Run it directly in VS Code (python main.py)
- Share it on GitHub as a starting point for your group
"""

import random
import copy
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
from tqdm import tqdm


# -----------------------------
# 1. Simple MLP model (proxy for an LLM)
# -----------------------------

class SimpleMLP(nn.Module):
    """
    A very small fully-connected network for MNIST.
    This is *not* an LLM, but it allows us to:
    - demonstrate pruning
    - run many experiments quickly (needed by the GA)
    The same pruning ideas transfer conceptually to LLMs.
    """
    def __init__(self, input_dim=28 * 28, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------
# 2. Data utilities
# -----------------------------

def get_mnist_dataloaders(batch_size: int = 128, train_subset: int = 10000, val_size: int = 2000):
    """
    Download MNIST and create small train/val loaders to keep GA evaluation fast.

    This is enough for the project because:
    - We only need a *relative* comparison between pruning strategies
    - Option A focuses more on analysis than on reaching SOTA accuracy
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # Use a subset of the training set for speed
    if train_subset is not None and train_subset < len(train_dataset):
        train_dataset, _ = random_split(train_dataset, [train_subset, len(train_dataset) - train_subset])

    # Split into train and validation
    val_size = min(val_size, len(train_dataset) // 5)
    train_size = len(train_dataset) - val_size
    train_subset_ds, val_subset_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# -----------------------------
# 3. Training and evaluation
# -----------------------------

def train_baseline(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 2):
    """
    Train a baseline model that will then be pruned.
    In your report, you can discuss:
    - training time
    - baseline accuracy
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={running_loss / len(train_loader):.4f}")


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate accuracy on the validation set (used as fitness by the GA).
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


# -----------------------------
# 4. Pruning utilities
# -----------------------------

def apply_global_pruning(model: nn.Module, pruning_rate: float) -> None:
    """
    Apply unstructured magnitude pruning to all linear layers of the model.

    pruning_rate: fraction of weights to prune (between 0.0 and 0.9 for example)
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    # After pruning, we can optionally remove the pruning re-parametrization
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")


def count_nonzero_params(model: nn.Module) -> Tuple[int, int]:
    nonzero = 0
    total = 0
    for param in model.parameters():
        if param is not None:
            nz = torch.count_nonzero(param).item()
            t = param.numel()
            nonzero += nz
            total += t
    return nonzero, total


# -----------------------------
# 5. Genetic Algorithm for pruning rate search
# -----------------------------

@dataclass
class Individual:
    pruning_rate: float  # between 0 and 0.9
    fitness: float = 0.0
    nonzero_params: int = 0
    total_params: int = 0


class GeneticAlgorithmPruning:
    """
    Simple Genetic Algorithm that searches over pruning_rate in [0, max_rate].

    This is the "AI decision-making algorithm" required by the project statement.
    """

    def __init__(
        self,
        base_model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        population_size: int = 8,
        generations: int = 5,
        max_pruning_rate: float = 0.9,
        mutation_std: float = 0.1,
        elitism: int = 2,
    ):
        self.base_model = base_model
        self.val_loader = val_loader
        self.device = device

        self.population_size = population_size
        self.generations = generations
        self.max_pruning_rate = max_pruning_rate
        self.mutation_std = mutation_std
        self.elitism = elitism  # number of best individuals copied to next generation

    def _evaluate_individual(self, individual: Individual) -> None:
        """
        Copy the baseline model, apply pruning, evaluate accuracy, and store fitness.
        """
        model_copy = copy.deepcopy(self.base_model)
        model_copy.to(self.device)

        # apply pruning
        apply_global_pruning(model_copy, individual.pruning_rate)

        # count sparsity for analysis
        nonzero, total = count_nonzero_params(model_copy)
        acc = evaluate(model_copy, self.val_loader, self.device)

        # Here, fitness is simply accuracy.
        # In your report, you can compare this to multi-objective fitness
        # that also penalizes large models.
        individual.fitness = acc
        individual.nonzero_params = nonzero
        individual.total_params = total

    def _init_population(self) -> List[Individual]:
        population = []
        for _ in range(self.population_size):
            rate = random.uniform(0.0, self.max_pruning_rate)
            population.append(Individual(pruning_rate=rate))
        return population

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """
        Tournament selection: pick 2*2 individuals at random and take the best of each pair.
        """
        def tournament():
            a, b = random.sample(population, 2)
            return a if a.fitness >= b.fitness else b

        return tournament(), tournament()

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Simple arithmetic crossover on pruning_rate.
        """
        alpha = random.random()
        child_rate = alpha * parent1.pruning_rate + (1 - alpha) * parent2.pruning_rate
        return Individual(pruning_rate=child_rate)

    def _mutate(self, individual: Individual) -> None:
        """
        Add Gaussian noise to pruning_rate, keep it within [0, max_pruning_rate].
        """
        new_rate = individual.pruning_rate + random.gauss(0.0, self.mutation_std)
        new_rate = max(0.0, min(self.max_pruning_rate, new_rate))
        individual.pruning_rate = new_rate

    def run(self) -> List[Individual]:
        """
        Main GA loop:
        - initialize population
        - for each generation: evaluate, select, crossover, mutate
        Returns the final population for analysis.
        """
        population = self._init_population()

        for gen in range(self.generations):
            print(f"\n=== Generation {gen+1}/{self.generations} ===")

            # Evaluate
            for ind in population:
                self._evaluate_individual(ind)

            # Sort by fitness (accuracy)
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            best = population[0]
            sparsity = 1.0 - best.nonzero_params / best.total_params
            print(
                f"Best individual: pruning_rate={best.pruning_rate:.3f}, "
                f"val_acc={best.fitness:.4f}, sparsity={sparsity:.3f}"
            )

            # Create next population with elitism
            next_population: List[Individual] = population[: self.elitism]

            # Fill the rest via crossover + mutation
            while len(next_population) < self.population_size:
                p1, p2 = self._select_parents(population)
                child = self._crossover(p1, p2)

                if random.random() < 0.8:  # mutation probability
                    self._mutate(child)

                next_population.append(child)

            population = next_population

        # Final evaluation to have stats for the last generation
        for ind in population:
            self._evaluate_individual(ind)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population


# -----------------------------
# 6. Main script
# -----------------------------

def main():
    # 1. Device selection (CPU is enough for this toy example)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Data
    train_loader, val_loader = get_mnist_dataloaders()

    # 3. Baseline model training
    baseline_model = SimpleMLP()
    print("Training baseline model...")
    train_baseline(baseline_model, train_loader, device, epochs=2)

    baseline_acc = evaluate(baseline_model, val_loader, device)
    nonzero, total = count_nonzero_params(baseline_model)
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print(f"Baseline non-zero params: {nonzero}/{total}")

    # 4. Genetic Algorithm to optimize pruning rate
    ga = GeneticAlgorithmPruning(
        base_model=baseline_model,
        val_loader=val_loader,
        device=device,
        population_size=8,
        generations=5,
        max_pruning_rate=0.9,
        mutation_std=0.1,
        elitism=2,
    )

    final_population = ga.run()

    # 5. Print best solution
    best = final_population[0]
    sparsity = 1.0 - best.nonzero_params / best.total_params
    print("\n=== Final Best Solution ===")
    print(f"Pruning rate: {best.pruning_rate:.3f}")
    print(f"Validation accuracy: {best.fitness:.4f}")
    print(f"Sparsity (1 - nonzero/total): {sparsity:.3f}")


if __name__ == "__main__":
    main()
