"""
Local Search algorithm for optimizing pruning hyperparameters.
"""

import numpy as np
from typing import Callable, Dict, Tuple, List, Any


class LocalSearchOptimizer:
    """
    Local Search optimizer for pruning strategies.
    
    Uses hill climbing and simulated annealing to find locally optimal
    pruning hyperparameters. This is efficient for fine-tuning parameters
    in a continuous search space.
    """
    
    def __init__(self,
                 max_iterations: int = 100,
                 initial_temperature: float = 1.0,
                 cooling_rate: float = 0.95,
                 step_size: float = 0.1,
                 use_simulated_annealing: bool = True):
        """
        Initialize Local Search optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            initial_temperature: Starting temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases
            step_size: Size of steps in parameter space
            use_simulated_annealing: Whether to use SA or simple hill climbing
        """
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.use_simulated_annealing = use_simulated_annealing
        self.history = []
        self.best_solution = None
        self.best_score = float('-inf')
    
    def generate_neighbor(self, 
                         current: Dict[str, float],
                         param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Generate a neighboring solution by perturbing current solution.
        
        Args:
            current: Current solution
            param_ranges: Valid parameter ranges
            
        Returns:
            Neighboring solution
        """
        neighbor = current.copy()
        
        # Randomly select a parameter to perturb
        param_to_change = np.random.choice(list(current.keys()))
        min_val, max_val = param_ranges[param_to_change]
        
        # Add random perturbation
        perturbation = np.random.uniform(-self.step_size, self.step_size) * (max_val - min_val)
        new_value = current[param_to_change] + perturbation
        
        # Clip to valid range
        neighbor[param_to_change] = np.clip(new_value, min_val, max_val)
        
        return neighbor
    
    def acceptance_probability(self, current_score: float, 
                              new_score: float, 
                              temperature: float) -> float:
        """
        Calculate probability of accepting a worse solution (for SA).
        
        Args:
            current_score: Current solution score
            new_score: New solution score
            temperature: Current temperature
            
        Returns:
            Acceptance probability
        """
        if new_score > current_score:
            return 1.0
        
        if temperature <= 0:
            return 0.0
        
        # Exponential acceptance probability
        delta = new_score - current_score
        return np.exp(delta / temperature)
    
    def optimize(self,
                 param_ranges: Dict[str, Tuple[float, float]],
                 objective_function: Callable[[Dict[str, float]], float],
                 initial_solution: Dict[str, float] = None) -> Dict[str, float]:
        """
        Run local search optimization.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            objective_function: Function to maximize
            initial_solution: Starting point (random if None)
            
        Returns:
            Best solution found
        """
        # Initialize solution
        if initial_solution is None:
            current = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                current[param_name] = np.random.uniform(min_val, max_val)
        else:
            current = initial_solution.copy()
        
        current_score = objective_function(current)
        
        # Track best solution
        self.best_solution = current.copy()
        self.best_score = current_score
        
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self.generate_neighbor(current, param_ranges)
            neighbor_score = objective_function(neighbor)
            
            # Decide whether to accept neighbor
            if self.use_simulated_annealing:
                # Simulated annealing
                accept_prob = self.acceptance_probability(
                    current_score, neighbor_score, temperature
                )
                accept = np.random.random() < accept_prob
            else:
                # Simple hill climbing
                accept = neighbor_score > current_score
            
            if accept:
                current = neighbor
                current_score = neighbor_score
            
            # Update best solution
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_solution = current.copy()
            
            # Cool down temperature
            if self.use_simulated_annealing:
                temperature *= self.cooling_rate
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'current_score': current_score,
                'best_score': self.best_score,
                'temperature': temperature if self.use_simulated_annealing else 0.0
            })
        
        return self.best_solution
    
    def multi_start_optimize(self,
                            param_ranges: Dict[str, Tuple[float, float]],
                            objective_function: Callable[[Dict[str, float]], float],
                            num_starts: int = 5) -> Dict[str, float]:
        """
        Run local search from multiple random starting points.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            objective_function: Function to maximize
            num_starts: Number of random starts
            
        Returns:
            Best solution found across all starts
        """
        global_best = None
        global_best_score = float('-inf')
        
        for start in range(num_starts):
            # Random initialization
            initial = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                initial[param_name] = np.random.uniform(min_val, max_val)
            
            # Run optimization
            solution = self.optimize(param_ranges, objective_function, initial)
            score = objective_function(solution)
            
            # Update global best
            if score > global_best_score:
                global_best_score = score
                global_best = solution
        
        return global_best
