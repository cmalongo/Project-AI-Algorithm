"""
Tests for pruning methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from src.pruning import MagnitudePruner, StructuredPruner


class TestMagnitudePruner(unittest.TestCase):
    """Test magnitude-based pruning."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_weights = np.random.randn(10, 10)
        self.test_model = {
            'layer1': np.random.randn(5, 5),
            'layer2': np.random.randn(5, 5)
        }
    
    def test_initialization(self):
        """Test pruner initialization."""
        pruner = MagnitudePruner(target_sparsity=0.5)
        self.assertEqual(pruner.target_sparsity, 0.5)
        self.assertTrue(pruner.preserve_accuracy)
    
    def test_invalid_sparsity(self):
        """Test that invalid sparsity raises error."""
        with self.assertRaises(ValueError):
            MagnitudePruner(target_sparsity=1.5)
        with self.assertRaises(ValueError):
            MagnitudePruner(target_sparsity=-0.1)
    
    def test_importance_scores(self):
        """Test importance score calculation."""
        pruner = MagnitudePruner(target_sparsity=0.5)
        scores = pruner.calculate_importance_scores(self.test_weights)
        
        # Scores should be absolute values
        self.assertTrue(np.all(scores >= 0))
        self.assertEqual(scores.shape, self.test_weights.shape)
    
    def test_prune_layer(self):
        """Test single layer pruning."""
        pruner = MagnitudePruner(target_sparsity=0.5)
        pruned = pruner.prune_layer(self.test_weights, 'test_layer')
        
        # Check sparsity
        sparsity = np.sum(pruned == 0) / pruned.size
        self.assertAlmostEqual(sparsity, 0.5, delta=0.1)
        
        # Check shape preserved
        self.assertEqual(pruned.shape, self.test_weights.shape)
    
    def test_prune_model(self):
        """Test full model pruning."""
        pruner = MagnitudePruner(target_sparsity=0.3)
        pruned_model = pruner.prune_model(self.test_model)
        
        # Check all layers pruned
        self.assertEqual(len(pruned_model), len(self.test_model))
        for layer_name in self.test_model:
            self.assertIn(layer_name, pruned_model)
            # Check some zeros exist
            self.assertTrue(np.sum(pruned_model[layer_name] == 0) > 0)


class TestStructuredPruner(unittest.TestCase):
    """Test structured pruning."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_weights = np.random.randn(10, 10)
    
    def test_initialization(self):
        """Test structured pruner initialization."""
        pruner = StructuredPruner(target_sparsity=0.5, prune_dimension=0)
        self.assertEqual(pruner.target_sparsity, 0.5)
        self.assertEqual(pruner.prune_dimension, 0)
    
    def test_row_pruning(self):
        """Test pruning entire rows."""
        pruner = StructuredPruner(target_sparsity=0.3, prune_dimension=0)
        pruned = pruner.prune_layer(self.test_weights, 'test_layer')
        
        # Check that some entire rows are zero
        zero_rows = np.all(pruned == 0, axis=1)
        self.assertTrue(np.any(zero_rows))
        
        # Verify approximately correct number of rows pruned
        num_zero_rows = np.sum(zero_rows)
        expected_rows = int(0.3 * self.test_weights.shape[0])
        self.assertAlmostEqual(num_zero_rows, expected_rows, delta=2)
    
    def test_column_pruning(self):
        """Test pruning entire columns."""
        pruner = StructuredPruner(target_sparsity=0.3, prune_dimension=1)
        pruned = pruner.prune_layer(self.test_weights, 'test_layer')
        
        # Check that some entire columns are zero
        zero_cols = np.all(pruned == 0, axis=0)
        self.assertTrue(np.any(zero_cols))
    
    def test_importance_calculation(self):
        """Test importance score calculation for structures."""
        pruner = StructuredPruner(target_sparsity=0.5, prune_dimension=0)
        importance = pruner.calculate_importance_scores(self.test_weights)
        
        # Should have one score per row
        self.assertEqual(len(importance), self.test_weights.shape[0])
        # All scores should be non-negative (L2 norms)
        self.assertTrue(np.all(importance >= 0))


if __name__ == '__main__':
    unittest.main()
