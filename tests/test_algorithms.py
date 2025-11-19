"""
Tests for AI optimization algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from src.algorithms import GeneticAlgorithm, LocalSearchOptimizer, ReinforcementLearningOptimizer


class TestGeneticAlgorithm(unittest.TestCase):
    """Test genetic algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm(population_size=20, generations=10)
        self.assertEqual(ga.population_size, 20)
        self.assertEqual(ga.generations, 10)
    
    def test_population_initialization(self):
        """Test population initialization."""
        ga = GeneticAlgorithm(population_size=10)
        param_ranges = {'param1': (0.0, 1.0), 'param2': (0.0, 10.0)}
        population = ga.initialize_population(param_ranges)
        
        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertIn('param1', individual)
            self.assertIn('param2', individual)
            self.assertTrue(0.0 <= individual['param1'] <= 1.0)
            self.assertTrue(0.0 <= individual['param2'] <= 10.0)
    
    def test_optimize_simple(self):
        """Test optimization on a simple function."""
        ga = GeneticAlgorithm(population_size=20, generations=5)
        
        # Simple fitness: maximize x
        def fitness(params):
            return params['x']
        
        param_ranges = {'x': (0.0, 10.0)}
        best = ga.optimize(param_ranges, fitness)
        
        # Should find value close to maximum
        self.assertGreater(best['x'], 7.0)
        self.assertIsNotNone(ga.best_individual)
        self.assertGreater(ga.best_fitness, 0)


class TestLocalSearchOptimizer(unittest.TestCase):
    """Test local search optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = LocalSearchOptimizer(max_iterations=50)
        self.assertEqual(optimizer.max_iterations, 50)
    
    def test_generate_neighbor(self):
        """Test neighbor generation."""
        optimizer = LocalSearchOptimizer()
        current = {'param1': 0.5, 'param2': 5.0}
        param_ranges = {'param1': (0.0, 1.0), 'param2': (0.0, 10.0)}
        
        neighbor = optimizer.generate_neighbor(current, param_ranges)
        
        # Neighbor should be different but within bounds
        self.assertIn('param1', neighbor)
        self.assertIn('param2', neighbor)
        self.assertTrue(0.0 <= neighbor['param1'] <= 1.0)
        self.assertTrue(0.0 <= neighbor['param2'] <= 10.0)
    
    def test_acceptance_probability(self):
        """Test acceptance probability calculation."""
        optimizer = LocalSearchOptimizer()
        
        # Better solution always accepted
        prob = optimizer.acceptance_probability(1.0, 2.0, 1.0)
        self.assertEqual(prob, 1.0)
        
        # Worse solution has lower probability
        prob = optimizer.acceptance_probability(2.0, 1.0, 1.0)
        self.assertLess(prob, 1.0)
    
    def test_optimize(self):
        """Test optimization."""
        optimizer = LocalSearchOptimizer(max_iterations=20)
        
        # Optimize to find maximum of quadratic
        def objective(params):
            x = params['x']
            return -(x - 5.0)**2  # Maximum at x=5
        
        param_ranges = {'x': (0.0, 10.0)}
        best = optimizer.optimize(param_ranges, objective)
        
        # Should find value close to 5
        self.assertAlmostEqual(best['x'], 5.0, delta=1.0)


class TestReinforcementLearningOptimizer(unittest.TestCase):
    """Test RL optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_initialization(self):
        """Test RL optimizer initialization."""
        rl = ReinforcementLearningOptimizer(
            state_dim=5, action_dim=3, episodes=10
        )
        self.assertEqual(rl.state_dim, 5)
        self.assertEqual(rl.action_dim, 3)
        self.assertEqual(rl.episodes, 10)
    
    def test_state_encoding(self):
        """Test state encoding."""
        rl = ReinforcementLearningOptimizer(state_dim=5)
        layer_info = {
            'size': 1000000,
            'sparsity': 0.5,
            'position': 0.3
        }
        state = rl.encode_state(layer_info)
        
        self.assertEqual(len(state), 5)
        self.assertTrue(np.all(np.isfinite(state)))
    
    def test_action_decoding(self):
        """Test action decoding."""
        rl = ReinforcementLearningOptimizer(action_dim=5)
        
        for action in range(5):
            sparsity = rl.decode_action(action)
            self.assertTrue(0.0 <= sparsity <= 1.0)
    
    def test_q_value_operations(self):
        """Test Q-value get and update."""
        rl = ReinforcementLearningOptimizer(state_dim=3, action_dim=3)
        state = np.array([0.5, 0.3, 0.2])
        
        # Initial Q-value should be 0
        q_val = rl.get_q_value(state, 0)
        self.assertEqual(q_val, 0.0)
        
        # Update Q-value
        next_state = np.array([0.6, 0.4, 0.1])
        rl.update_q_value(state, 0, 1.0, next_state)
        
        # Q-value should have changed
        new_q_val = rl.get_q_value(state, 0)
        self.assertNotEqual(new_q_val, 0.0)
    
    def test_action_selection(self):
        """Test action selection."""
        rl = ReinforcementLearningOptimizer(state_dim=3, action_dim=3)
        state = np.array([0.5, 0.3, 0.2])
        
        # Should return valid action
        action = rl.select_action(state, explore=False)
        self.assertTrue(0 <= action < 3)


if __name__ == '__main__':
    unittest.main()
