"""
Tests for utility functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from src.utils.metrics import (
    calculate_sparsity, calculate_compression_ratio,
    evaluate_accuracy_loss, evaluate_pruning_quality
)
from src.utils.hardware import (
    estimate_inference_speedup, calculate_memory_reduction,
    estimate_energy_savings
)


class TestMetrics(unittest.TestCase):
    """Test metric calculations."""
    
    def test_calculate_sparsity(self):
        """Test sparsity calculation."""
        # 50% zeros
        weights = np.array([[1, 0], [0, 1]])
        sparsity = calculate_sparsity(weights)
        self.assertEqual(sparsity, 0.5)
        
        # All zeros
        weights = np.zeros((5, 5))
        sparsity = calculate_sparsity(weights)
        self.assertEqual(sparsity, 1.0)
        
        # No zeros
        weights = np.ones((5, 5))
        sparsity = calculate_sparsity(weights)
        self.assertEqual(sparsity, 0.0)
    
    def test_calculate_compression_ratio(self):
        """Test compression ratio calculation."""
        original = {'layer1': np.ones((10, 10))}
        
        # 50% pruned
        pruned = {'layer1': np.array([[1, 0]] * 50).reshape(10, 10)}
        ratio = calculate_compression_ratio(original, pruned)
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)
    
    def test_evaluate_accuracy_loss(self):
        """Test accuracy loss calculation."""
        loss = evaluate_accuracy_loss(0.9, 0.85)
        self.assertAlmostEqual(loss, 0.05, places=5)
        
        # Negative loss = improvement
        loss = evaluate_accuracy_loss(0.8, 0.85)
        self.assertAlmostEqual(loss, -0.05, places=5)
    
    def test_evaluate_pruning_quality(self):
        """Test comprehensive quality evaluation."""
        original = {'layer1': np.ones((10, 10))}
        pruned = {'layer1': np.array([[1, 0]] * 50).reshape(10, 10)}
        
        metrics = evaluate_pruning_quality(original, pruned, 0.9, 0.85)
        
        self.assertIn('compression_ratio', metrics)
        self.assertIn('sparsity', metrics)
        self.assertIn('accuracy_loss', metrics)
        self.assertIn('accuracy_retention', metrics)
        
        self.assertGreater(metrics['compression_ratio'], 1.0)
        self.assertAlmostEqual(metrics['sparsity'], 0.5, delta=0.01)


class TestHardwareUtils(unittest.TestCase):
    """Test hardware utility functions."""
    
    def test_estimate_inference_speedup(self):
        """Test speedup estimation."""
        original = {'layer1': np.ones((10, 10))}
        pruned = {'layer1': np.array([[1, 0]] * 50).reshape(10, 10)}
        
        speedup = estimate_inference_speedup(original, pruned, hardware_type="cpu")
        self.assertGreater(speedup, 1.0)
        
        # GPU should have different speedup
        speedup_gpu = estimate_inference_speedup(original, pruned, hardware_type="gpu")
        self.assertGreater(speedup_gpu, 1.0)
    
    def test_calculate_memory_reduction(self):
        """Test memory reduction calculation."""
        original = {'layer1': np.ones((100, 100))}
        # 80% pruned
        pruned_data = np.ones((100, 100))
        pruned_data[:80, :] = 0
        pruned = {'layer1': pruned_data}
        
        memory = calculate_memory_reduction(original, pruned, use_sparse_storage=True)
        
        self.assertIn('original_size_mb', memory)
        self.assertIn('pruned_size_mb', memory)
        self.assertIn('reduction_ratio', memory)
        
        self.assertGreater(memory['reduction_ratio'], 0.0)
        self.assertLess(memory['pruned_size_mb'], memory['original_size_mb'])
    
    def test_estimate_energy_savings(self):
        """Test energy savings estimation."""
        # 50% sparsity
        savings = estimate_energy_savings(0.5, hardware_type="cpu")
        self.assertGreater(savings, 0.0)
        self.assertLessEqual(savings, 1.0)
        
        # Higher sparsity = more savings
        savings_high = estimate_energy_savings(0.8, hardware_type="cpu")
        self.assertGreater(savings_high, savings)


if __name__ == '__main__':
    unittest.main()
