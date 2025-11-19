"""
Magnitude-based pruning implementation.
"""

import numpy as np
from .base_pruner import BasePruner


class MagnitudePruner(BasePruner):
    """
    Magnitude-based pruning removes parameters with smallest absolute values.
    
    This is one of the simplest and most effective pruning methods, based on
    the intuition that parameters with small magnitudes contribute less to
    the model's output.
    """
    
    def calculate_importance_scores(self, layer_weights: np.ndarray) -> np.ndarray:
        """
        Calculate importance scores based on absolute magnitude.
        
        Args:
            layer_weights: Weight matrix of the layer
            
        Returns:
            Absolute values of weights as importance scores
        """
        return np.abs(layer_weights)
    
    def prune_layer(self, layer_weights: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Prune layer by zeroing out smallest magnitude weights.
        
        Args:
            layer_weights: Weight matrix of the layer
            layer_name: Name of the layer
            
        Returns:
            Pruned weight matrix
        """
        # Calculate importance scores
        importance = self.calculate_importance_scores(layer_weights)
        
        # Flatten for easier manipulation
        flat_importance = importance.flatten()
        flat_weights = layer_weights.flatten()
        
        # Calculate threshold for pruning
        threshold_idx = int(self.target_sparsity * len(flat_importance))
        
        if threshold_idx > 0:
            # Sort and find threshold value
            sorted_importance = np.sort(flat_importance)
            threshold_value = sorted_importance[threshold_idx]
            
            # Create mask: keep weights above threshold
            mask = importance > threshold_value
            
            # Apply mask
            pruned_weights = layer_weights * mask
        else:
            pruned_weights = layer_weights.copy()
        
        return pruned_weights
