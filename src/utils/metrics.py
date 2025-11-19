"""
Metrics for evaluating model compression.
"""

import numpy as np
from typing import Dict, Callable, Any


def calculate_sparsity(weights: np.ndarray) -> float:
    """
    Calculate sparsity of a weight matrix.
    
    Sparsity is the fraction of zero weights.
    
    Args:
        weights: Weight matrix
        
    Returns:
        Sparsity ratio (0.0 to 1.0)
    """
    if weights.size == 0:
        return 0.0
    
    zero_count = np.sum(weights == 0)
    total_count = weights.size
    
    return zero_count / total_count


def calculate_compression_ratio(original_weights: Dict[str, np.ndarray],
                                pruned_weights: Dict[str, np.ndarray]) -> float:
    """
    Calculate overall compression ratio.
    
    Args:
        original_weights: Original model weights
        pruned_weights: Pruned model weights
        
    Returns:
        Compression ratio (>1 means compression achieved)
    """
    original_size = sum(w.size for w in original_weights.values())
    
    # Count non-zero parameters in pruned model
    pruned_nonzero = sum(np.count_nonzero(w) for w in pruned_weights.values())
    
    if pruned_nonzero == 0:
        return float('inf')
    
    return original_size / pruned_nonzero


def evaluate_accuracy_loss(original_accuracy: float,
                           pruned_accuracy: float) -> float:
    """
    Calculate accuracy loss from pruning.
    
    Args:
        original_accuracy: Accuracy of original model (0.0 to 1.0)
        pruned_accuracy: Accuracy of pruned model (0.0 to 1.0)
        
    Returns:
        Accuracy loss (negative means improvement)
    """
    return original_accuracy - pruned_accuracy


def calculate_model_size(weights: Dict[str, np.ndarray]) -> int:
    """
    Calculate total number of parameters.
    
    Args:
        weights: Model weights
        
    Returns:
        Total parameter count
    """
    return sum(w.size for w in weights.values())


def calculate_nonzero_params(weights: Dict[str, np.ndarray]) -> int:
    """
    Calculate number of non-zero parameters.
    
    Args:
        weights: Model weights
        
    Returns:
        Non-zero parameter count
    """
    return sum(np.count_nonzero(w) for w in weights.values())


def evaluate_pruning_quality(original_weights: Dict[str, np.ndarray],
                             pruned_weights: Dict[str, np.ndarray],
                             original_accuracy: float,
                             pruned_accuracy: float) -> Dict[str, float]:
    """
    Comprehensive evaluation of pruning quality.
    
    Args:
        original_weights: Original model weights
        pruned_weights: Pruned model weights
        original_accuracy: Original model accuracy
        pruned_accuracy: Pruned model accuracy
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Compression metrics
    metrics['compression_ratio'] = calculate_compression_ratio(
        original_weights, pruned_weights
    )
    metrics['sparsity'] = np.mean([
        calculate_sparsity(w) for w in pruned_weights.values()
    ])
    
    # Accuracy metrics
    metrics['accuracy_loss'] = evaluate_accuracy_loss(
        original_accuracy, pruned_accuracy
    )
    metrics['accuracy_retention'] = pruned_accuracy / original_accuracy if original_accuracy > 0 else 0.0
    
    # Size metrics
    metrics['original_size'] = calculate_model_size(original_weights)
    metrics['pruned_size'] = calculate_nonzero_params(pruned_weights)
    metrics['size_reduction'] = 1.0 - (metrics['pruned_size'] / metrics['original_size'])
    
    return metrics
