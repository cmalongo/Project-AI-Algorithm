"""
Structured pruning implementation for hardware-friendly compression.
"""

import numpy as np
from .base_pruner import BasePruner


class StructuredPruner(BasePruner):
    """
    Structured pruning removes entire neurons, channels, or filters.
    
    This approach is more hardware-friendly than unstructured pruning as it
    doesn't require sparse matrix operations. It removes entire structural
    units which can directly reduce computational requirements.
    """
    
    def __init__(self, target_sparsity: float = 0.5, 
                 preserve_accuracy: bool = True,
                 prune_dimension: int = 0):
        """
        Initialize structured pruner.
        
        Args:
            target_sparsity: Fraction of structures to prune (0.0 to 1.0)
            preserve_accuracy: Whether to prioritize accuracy preservation
            prune_dimension: Dimension along which to prune (0 for rows, 1 for columns)
        """
        super().__init__(target_sparsity, preserve_accuracy)
        self.prune_dimension = prune_dimension
    
    def calculate_importance_scores(self, layer_weights: np.ndarray) -> np.ndarray:
        """
        Calculate importance scores for structures (rows/columns).
        
        Args:
            layer_weights: Weight matrix of the layer
            
        Returns:
            Importance scores for each structure (L2 norm)
        """
        # Calculate L2 norm along the pruning dimension
        importance = np.linalg.norm(layer_weights, axis=1-self.prune_dimension)
        return importance
    
    def prune_layer(self, layer_weights: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Prune entire structures (rows or columns) from the layer.
        
        Args:
            layer_weights: Weight matrix of the layer
            layer_name: Name of the layer
            
        Returns:
            Pruned weight matrix with some structures zeroed out
        """
        # Calculate importance for each structure
        importance = self.calculate_importance_scores(layer_weights)
        
        # Calculate number of structures to prune
        num_structures = len(importance)
        num_to_prune = int(self.target_sparsity * num_structures)
        
        if num_to_prune > 0:
            # Find indices of least important structures
            prune_indices = np.argsort(importance)[:num_to_prune]
            
            # Create pruned copy
            pruned_weights = layer_weights.copy()
            
            # Zero out entire structures
            if self.prune_dimension == 0:
                # Prune rows (output neurons)
                pruned_weights[prune_indices, :] = 0
            else:
                # Prune columns (input neurons)
                pruned_weights[:, prune_indices] = 0
        else:
            pruned_weights = layer_weights.copy()
        
        return pruned_weights
