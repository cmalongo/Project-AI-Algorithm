"""
Pruning methods for model compression.
"""

from .base_pruner import BasePruner
from .magnitude_pruner import MagnitudePruner
from .structured_pruner import StructuredPruner

__all__ = ["BasePruner", "MagnitudePruner", "StructuredPruner"]
