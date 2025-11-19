"""
Reinforcement Learning for optimizing pruning strategies.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Any


class ReinforcementLearningOptimizer:
    """
    Q-Learning based optimizer for pruning strategies.
    
    Uses reinforcement learning to learn optimal pruning decisions by
    treating pruning as a sequential decision-making process where
    the agent learns which layers and sparsity levels to prune.
    """
    
    def __init__(self,
                 state_dim: int = 10,
                 action_dim: int = 5,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1,
                 episodes: int = 100):
        """
        Initialize RL optimizer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate
            episodes: Number of training episodes
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        
        # Initialize Q-table
        self.q_table = {}
        self.history = []
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state array to hashable key.
        
        Args:
            state: State vector
            
        Returns:
            String key for Q-table
        """
        # Discretize continuous state
        discretized = np.round(state, decimals=2)
        return str(discretized.tolist())
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: Current state
            action: Action index
            
        Returns:
            Q-value
        """
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        return self.q_table[key][action]
    
    def update_q_value(self, state: np.ndarray, action: int, 
                      reward: float, next_state: np.ndarray):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)
        
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_dim)
        
        # Q-learning update
        current_q = self.q_table[key][action]
        max_next_q = np.max(self.q_table[next_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[key][action] = new_q
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration
            
        Returns:
            Selected action index
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploit: best known action
            key = self._state_to_key(state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_dim)
            return np.argmax(self.q_table[key])
    
    def encode_state(self, layer_info: Dict[str, Any]) -> np.ndarray:
        """
        Encode layer information into state vector.
        
        Args:
            layer_info: Dictionary with layer properties
            
        Returns:
            State vector
        """
        # Extract features from layer info
        features = []
        
        # Normalize layer size
        if 'size' in layer_info:
            features.append(layer_info['size'] / 1e6)  # Normalize by 1M
        else:
            features.append(0.0)
        
        # Current sparsity
        if 'sparsity' in layer_info:
            features.append(layer_info['sparsity'])
        else:
            features.append(0.0)
        
        # Layer position (normalized)
        if 'position' in layer_info:
            features.append(layer_info['position'])
        else:
            features.append(0.0)
        
        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def decode_action(self, action: int) -> float:
        """
        Decode action index to sparsity level.
        
        Args:
            action: Action index
            
        Returns:
            Sparsity level (0.0 to 1.0)
        """
        # Map actions to sparsity levels
        sparsity_levels = np.linspace(0.0, 0.9, self.action_dim)
        return sparsity_levels[action]
    
    def train(self, 
              environment_step: Callable[[np.ndarray, int], Tuple[np.ndarray, float, bool]]):
        """
        Train the RL agent.
        
        Args:
            environment_step: Function that takes (state, action) and returns 
                            (next_state, reward, done)
        """
        for episode in range(self.episodes):
            # Initialize episode
            state = np.random.rand(self.state_dim)
            episode_reward = 0
            steps = 0
            max_steps = 50
            
            while steps < max_steps:
                # Select and execute action
                action = self.select_action(state, explore=True)
                next_state, reward, done = environment_step(state, action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            self.history.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': steps
            })
    
    def get_optimal_policy(self, layer_info: Dict[str, Any]) -> float:
        """
        Get optimal pruning sparsity for a layer.
        
        Args:
            layer_info: Layer information
            
        Returns:
            Recommended sparsity level
        """
        state = self.encode_state(layer_info)
        action = self.select_action(state, explore=False)
        return self.decode_action(action)
