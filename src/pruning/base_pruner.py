"""
Base class for all pruning methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BasePruner(ABC):
    """
    Abstract base class for pruning methods.
    
    Pruning removes less important parameters from neural networks to achieve
    model compression while preserving accuracy as much as possible.
    """
    
    def __init__(self, target_sparsity: float = 0.5, preserve_accuracy: bool = True):
        """
        Initialize the pruner.
        
        Args:
            target_sparsity: Fraction of parameters to prune (0.0 to 1.0)
            preserve_accuracy: Whether to prioritize accuracy preservation
        """
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be between 0.0 and 1.0")
        
        self.target_sparsity = target_sparsity
        self.preserve_accuracy = preserve_accuracy
        self.pruned_layers = {}
        self.original_params = {}
        
    @abstractmethod
    def calculate_importance_scores(self, layer_weights: np.ndarray) -> np.ndarray:
        """
        Calculate importance scores for each parameter.
        
        Args:
            layer_weights: Weight matrix of the layer
            
        Returns:
            Importance scores for each parameter
        """
        pass
    
    @abstractmethod
    def prune_layer(self, layer_weights: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Prune a single layer based on importance scores.
        
        Args:
            layer_weights: Weight matrix of the layer
            layer_name: Name of the layer
            
        Returns:
            Pruned weight matrix
        """
        pass
    
    def prune_model(self, model_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prune an entire model.
        
        Args:
            model_weights: Dictionary mapping layer names to weight matrices
            
        Returns:
            Dictionary of pruned layer weights
        """
        self.original_params = model_weights.copy()
        pruned_weights = {}
        
        for layer_name, weights in model_weights.items():
            pruned_weights[layer_name] = self.prune_layer(weights, layer_name)
            self.pruned_layers[layer_name] = True
            
        return pruned_weights
    
    def get_sparsity_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the pruning operation.
        
        Returns:
            Dictionary containing sparsity statistics
        """
        stats = {}
        for layer_name in self.pruned_layers:
            if layer_name in self.original_params:
                original = self.original_params[layer_name]
                total_params = original.size
                zero_params = np.sum(original == 0)
                stats[layer_name] = zero_params / total_params if total_params > 0 else 0.0
        
        return stats
