"""
Basic demonstration of LLM pruning with AI optimization algorithms.

This example shows how to:
1. Create a simple model representation
2. Apply different pruning methods
3. Use AI algorithms to optimize pruning hyperparameters
4. Evaluate compression results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.pruning import MagnitudePruner, StructuredPruner
from src.algorithms import GeneticAlgorithm, LocalSearchOptimizer, ReinforcementLearningOptimizer
from src.utils.metrics import evaluate_pruning_quality, calculate_sparsity
from src.utils.hardware import estimate_inference_speedup, calculate_memory_reduction


def create_dummy_model(num_layers: int = 3, layer_size: int = 100) -> dict:
    """Create a dummy model for demonstration."""
    model = {}
    for i in range(num_layers):
        # Create random weight matrices
        weights = np.random.randn(layer_size, layer_size) * 0.1
        model[f'layer_{i}'] = weights
    return model


def simulate_model_accuracy(weights: dict, reference_weights: dict = None) -> float:
    """
    Simulate model accuracy based on weight similarity.
    
    In real scenarios, this would evaluate the model on a validation set.
    """
    if reference_weights is None:
        # Assume original model has 90% accuracy
        return 0.90
    
    # Calculate how much weights deviate from original
    total_diff = 0
    total_params = 0
    
    for layer_name in weights:
        if layer_name in reference_weights:
            diff = np.sum(np.abs(weights[layer_name] - reference_weights[layer_name]))
            total_diff += diff
            total_params += weights[layer_name].size
    
    # More deviation = lower accuracy (simplified)
    avg_diff = total_diff / total_params if total_params > 0 else 0
    simulated_accuracy = 0.90 - min(avg_diff * 10, 0.30)  # Cap loss at 30%
    
    return max(simulated_accuracy, 0.0)


def demo_magnitude_pruning():
    """Demonstrate magnitude-based pruning."""
    print("\n" + "="*70)
    print("DEMO 1: Magnitude-Based Pruning")
    print("="*70)
    
    # Create model
    model = create_dummy_model(num_layers=3, layer_size=50)
    print(f"\nCreated model with {len(model)} layers")
    
    # Apply pruning
    pruner = MagnitudePruner(target_sparsity=0.5)
    pruned_model = pruner.prune_model(model)
    
    # Evaluate results
    original_acc = simulate_model_accuracy(model)
    pruned_acc = simulate_model_accuracy(pruned_model, model)
    
    metrics = evaluate_pruning_quality(model, pruned_model, original_acc, pruned_acc)
    
    print(f"\nPruning Results:")
    print(f"  Sparsity: {metrics['sparsity']:.2%}")
    print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  Original Accuracy: {original_acc:.2%}")
    print(f"  Pruned Accuracy: {pruned_acc:.2%}")
    print(f"  Accuracy Loss: {metrics['accuracy_loss']:.2%}")
    
    # Hardware benefits
    speedup = estimate_inference_speedup(model, pruned_model)
    memory = calculate_memory_reduction(model, pruned_model)
    
    print(f"\nHardware Benefits:")
    print(f"  Estimated Speedup: {speedup:.2f}x")
    print(f"  Memory Reduction: {memory['reduction_ratio']:.2%}")
    print(f"  Size: {memory['original_size_mb']:.2f}MB → {memory['pruned_size_mb']:.2f}MB")


def demo_structured_pruning():
    """Demonstrate structured pruning for hardware efficiency."""
    print("\n" + "="*70)
    print("DEMO 2: Structured Pruning (Hardware-Friendly)")
    print("="*70)
    
    # Create model
    model = create_dummy_model(num_layers=3, layer_size=50)
    
    # Apply structured pruning
    pruner = StructuredPruner(target_sparsity=0.3, prune_dimension=0)
    pruned_model = pruner.prune_model(model)
    
    # Evaluate
    original_acc = simulate_model_accuracy(model)
    pruned_acc = simulate_model_accuracy(pruned_model, model)
    
    metrics = evaluate_pruning_quality(model, pruned_model, original_acc, pruned_acc)
    
    print(f"\nStructured Pruning Results:")
    print(f"  Sparsity: {metrics['sparsity']:.2%}")
    print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  Accuracy Retention: {metrics['accuracy_retention']:.2%}")
    
    print("\nNote: Structured pruning is more hardware-friendly because it removes")
    print("      entire neurons/channels, avoiding sparse matrix operations.")


def demo_genetic_algorithm_optimization():
    """Demonstrate genetic algorithm for hyperparameter optimization."""
    print("\n" + "="*70)
    print("DEMO 3: Genetic Algorithm for Hyperparameter Optimization")
    print("="*70)
    
    # Create model
    model = create_dummy_model(num_layers=3, layer_size=50)
    
    # Define fitness function
    def fitness_function(params):
        """Fitness balances compression and accuracy."""
        sparsity = params['sparsity']
        
        # Apply pruning with these parameters
        pruner = MagnitudePruner(target_sparsity=sparsity)
        pruned = pruner.prune_model(model)
        
        # Evaluate
        original_acc = simulate_model_accuracy(model)
        pruned_acc = simulate_model_accuracy(pruned, model)
        
        # Fitness: maximize compression while minimizing accuracy loss
        compression = calculate_sparsity(list(pruned.values())[0])
        accuracy_loss = original_acc - pruned_acc
        
        # Weighted combination
        fitness = compression * 0.7 - accuracy_loss * 3.0
        return fitness
    
    # Run genetic algorithm
    ga = GeneticAlgorithm(
        population_size=20,
        generations=10,
        mutation_rate=0.15,
        elite_size=3
    )
    
    param_ranges = {'sparsity': (0.1, 0.8)}
    
    print("\nRunning Genetic Algorithm...")
    best_params = ga.optimize(param_ranges, fitness_function)
    
    print(f"\nOptimization Results:")
    print(f"  Best Sparsity: {best_params['sparsity']:.2%}")
    print(f"  Best Fitness: {ga.best_fitness:.4f}")
    print(f"  Generations: {ga.generations}")


def demo_local_search_optimization():
    """Demonstrate local search for parameter tuning."""
    print("\n" + "="*70)
    print("DEMO 4: Local Search Optimization")
    print("="*70)
    
    # Create model
    model = create_dummy_model(num_layers=2, layer_size=50)
    
    # Define objective function
    def objective_function(params):
        sparsity = params['sparsity']
        
        pruner = MagnitudePruner(target_sparsity=sparsity)
        pruned = pruner.prune_model(model)
        
        original_acc = simulate_model_accuracy(model)
        pruned_acc = simulate_model_accuracy(pruned, model)
        
        # Balance compression and accuracy
        compression_score = sparsity
        accuracy_penalty = (original_acc - pruned_acc) * 2.0
        
        return compression_score - accuracy_penalty
    
    # Run local search with simulated annealing
    optimizer = LocalSearchOptimizer(
        max_iterations=50,
        initial_temperature=0.5,
        cooling_rate=0.95,
        use_simulated_annealing=True
    )
    
    param_ranges = {'sparsity': (0.2, 0.8)}
    
    print("\nRunning Local Search with Simulated Annealing...")
    best_params = optimizer.optimize(param_ranges, objective_function)
    
    print(f"\nOptimization Results:")
    print(f"  Best Sparsity: {best_params['sparsity']:.2%}")
    print(f"  Best Score: {optimizer.best_score:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("LLM Model Compression Framework - Demonstration")
    print("="*70)
    print("\nThis framework addresses the key challenges in model compression:")
    print("  (i)   Preserving accuracy while removing parameters")
    print("  (ii)  Hardware-friendly compression strategies")
    print("  (iii) Hyperparameter optimization")
    
    # Run demos
    demo_magnitude_pruning()
    demo_structured_pruning()
    demo_genetic_algorithm_optimization()
    demo_local_search_optimization()
    
    print("\n" + "="*70)
    print("Demonstration Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Multiple pruning strategies available (magnitude, structured)")
    print("  • AI algorithms optimize hyperparameters automatically")
    print("  • Framework supports hardware-aware compression")
    print("  • Balances accuracy preservation with compression ratio")
    print()


if __name__ == "__main__":
    main()
