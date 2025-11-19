"""
Genetic Algorithm for optimizing pruning hyperparameters.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing pruning strategies.
    
    Uses evolutionary principles (selection, crossover, mutation) to find
    optimal pruning hyperparameters that balance compression and accuracy.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_size: int = 5):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to keep unchanged
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.history = []
    
    def initialize_population(self, param_ranges: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """
        Initialize random population.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            
        Returns:
            List of individuals (parameter dictionaries)
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                individual[param_name] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Dict[str, float], 
                        fitness_function: Callable[[Dict[str, float]], float]) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Parameter dictionary
            fitness_function: Function that evaluates the individual
            
        Returns:
            Fitness score
        """
        return fitness_function(individual)
    
    def select_parents(self, population: List[Dict[str, float]], 
                      fitness_scores: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Select two parents using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            Two parent individuals
        """
        # Tournament selection
        tournament_size = 3
        
        def tournament():
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            return population[winner_idx]
        
        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2
    
    def crossover(self, parent1: Dict[str, float], 
                  parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1 = {}
        offspring2 = {}
        
        for key in parent1.keys():
            if np.random.random() > 0.5:
                offspring1[key] = parent1[key]
                offspring2[key] = parent2[key]
            else:
                offspring1[key] = parent2[key]
                offspring2[key] = parent1[key]
        
        return offspring1, offspring2
    
    def mutate(self, individual: Dict[str, float], 
               param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate
            param_ranges: Valid parameter ranges
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for param_name, value in mutated.items():
            if np.random.random() < self.mutation_rate:
                min_val, max_val = param_ranges[param_name]
                # Add Gaussian noise
                noise = np.random.normal(0, (max_val - min_val) * 0.1)
                mutated[param_name] = np.clip(value + noise, min_val, max_val)
        
        return mutated
    
    def optimize(self, 
                 param_ranges: Dict[str, Tuple[float, float]],
                 fitness_function: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        """
        Run genetic algorithm optimization.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            fitness_function: Function to maximize
            
        Returns:
            Best individual found
        """
        # Initialize population
        population = self.initialize_population(param_ranges)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ind, fitness_function) 
                            for ind in population]
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = population[best_idx].copy()
            
            self.history.append({
                'generation': generation,
                'best_fitness': fitness_scores[best_idx],
                'avg_fitness': np.mean(fitness_scores)
            })
            
            # Create next generation
            next_generation = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_generation.append(population[idx].copy())
            
            # Generate offspring
            while len(next_generation) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1, param_ranges)
                offspring2 = self.mutate(offspring2, param_ranges)
                
                next_generation.append(offspring1)
                if len(next_generation) < self.population_size:
                    next_generation.append(offspring2)
            
            population = next_generation
        
        return self.best_individual
