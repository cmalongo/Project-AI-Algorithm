"""
Hardware-related utilities for estimating performance benefits.
"""

import numpy as np
from typing import Dict


def estimate_inference_speedup(original_weights: Dict[str, np.ndarray],
                               pruned_weights: Dict[str, np.ndarray],
                               hardware_type: str = "cpu") -> float:
    """
    Estimate inference speedup from pruning.
    
    This is a simplified estimation. Real speedup depends on:
    - Hardware architecture
    - Sparse matrix support
    - Memory bandwidth
    - Cache efficiency
    
    Args:
        original_weights: Original model weights
        pruned_weights: Pruned model weights
        hardware_type: Target hardware ("cpu", "gpu", "edge")
        
    Returns:
        Estimated speedup factor
    """
    # Calculate sparsity
    total_params = sum(w.size for w in original_weights.values())
    nonzero_params = sum(np.count_nonzero(w) for w in pruned_weights.values())
    
    if nonzero_params == 0:
        return 1.0
    
    sparsity = 1.0 - (nonzero_params / total_params)
    
    # Hardware-specific speedup factors
    # These are rough estimates based on typical hardware characteristics
    if hardware_type == "cpu":
        # CPUs have limited sparse operation support
        # Speedup is roughly proportional to sparsity but with overhead
        base_speedup = 1.0 / (1.0 - sparsity * 0.7)
    elif hardware_type == "gpu":
        # GPUs have better sparse support but require structured sparsity
        # for significant speedup
        base_speedup = 1.0 / (1.0 - sparsity * 0.5)
    elif hardware_type == "edge":
        # Edge devices benefit most from reduced memory and computation
        base_speedup = 1.0 / (1.0 - sparsity * 0.8)
    else:
        # Default conservative estimate
        base_speedup = 1.0 / (1.0 - sparsity * 0.6)
    
    return base_speedup


def calculate_memory_reduction(original_weights: Dict[str, np.ndarray],
                               pruned_weights: Dict[str, np.ndarray],
                               use_sparse_storage: bool = True) -> Dict[str, float]:
    """
    Calculate memory reduction from pruning.
    
    Args:
        original_weights: Original model weights
        pruned_weights: Pruned model weights
        use_sparse_storage: Whether sparse storage format is used
        
    Returns:
        Dictionary with memory reduction statistics
    """
    # Calculate sizes (assuming float32 = 4 bytes)
    bytes_per_param = 4
    
    original_size = sum(w.size for w in original_weights.values()) * bytes_per_param
    
    if use_sparse_storage:
        # Sparse storage: store only non-zero values + indices
        # CSR format: values (4 bytes) + col_indices (4 bytes) + row_ptr (4 bytes)
        nonzero_count = sum(np.count_nonzero(w) for w in pruned_weights.values())
        row_count = sum(w.shape[0] for w in pruned_weights.values() if len(w.shape) > 1)
        
        # Approximate CSR storage size
        pruned_size = (nonzero_count * 8) + (row_count * 4)
    else:
        # Dense storage: store all parameters
        pruned_size = sum(w.size for w in pruned_weights.values()) * bytes_per_param
    
    reduction_bytes = original_size - pruned_size
    reduction_ratio = 1.0 - (pruned_size / original_size) if original_size > 0 else 0.0
    
    return {
        'original_size_mb': original_size / (1024 * 1024),
        'pruned_size_mb': pruned_size / (1024 * 1024),
        'reduction_mb': reduction_bytes / (1024 * 1024),
        'reduction_ratio': reduction_ratio,
        'compression_factor': original_size / pruned_size if pruned_size > 0 else float('inf')
    }


def estimate_energy_savings(sparsity: float, 
                           hardware_type: str = "cpu") -> float:
    """
    Estimate energy savings from pruning.
    
    Sparse operations can save energy by:
    - Skipping zero multiplications
    - Reducing memory transfers
    - Lowering cache misses
    
    Args:
        sparsity: Fraction of zero weights
        hardware_type: Target hardware type
        
    Returns:
        Estimated energy savings (0.0 to 1.0)
    """
    # Energy savings depend on hardware architecture
    if hardware_type == "cpu":
        # CPUs: energy savings from skipped operations
        base_savings = sparsity * 0.6
    elif hardware_type == "gpu":
        # GPUs: savings depend on memory bandwidth reduction
        base_savings = sparsity * 0.4
    elif hardware_type == "edge":
        # Edge: significant savings from reduced computation
        base_savings = sparsity * 0.7
    else:
        base_savings = sparsity * 0.5
    
    return min(base_savings, 1.0)


def is_hardware_friendly(pruned_weights: Dict[str, np.ndarray],
                        min_block_size: int = 4) -> Dict[str, bool]:
    """
    Check if pruning pattern is hardware-friendly.
    
    Hardware-friendly pruning patterns:
    - Block sparsity (groups of zeros)
    - Structured sparsity (entire rows/columns)
    - Regular patterns
    
    Args:
        pruned_weights: Pruned model weights
        min_block_size: Minimum block size for efficient execution
        
    Returns:
        Dictionary indicating hardware-friendliness for each layer
    """
    results = {}
    
    for layer_name, weights in pruned_weights.items():
        # Check for structured sparsity (entire rows/columns zero)
        if len(weights.shape) >= 2:
            row_sparsity = np.all(weights == 0, axis=1)
            col_sparsity = np.all(weights == 0, axis=0)
            
            has_structured = np.any(row_sparsity) or np.any(col_sparsity)
            results[layer_name] = has_structured
        else:
            # For 1D weights, check block sparsity
            results[layer_name] = False
    
    return results
