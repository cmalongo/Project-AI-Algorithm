# Project-AI-Algorithm
AI Algorithms – (Model Compression: Pruning)

## Overview

This project implements **model compression for Large Language Models (LLMs)** using pruning methods combined with AI decision-making algorithms. The framework addresses key challenges in LLM optimization:

1. **Preserving accuracy** while removing a large fraction of parameters
2. **Hardware-friendly compression** strategies
3. **Hyperparameter optimization** using AI algorithms

## Features

### Pruning Methods
- **Magnitude-based Pruning**: Removes parameters with smallest absolute values
- **Structured Pruning**: Removes entire neurons/channels for hardware efficiency

### AI Optimization Algorithms
- **Genetic Algorithm**: Evolutionary optimization of pruning hyperparameters
- **Reinforcement Learning**: Q-learning based strategy optimization
- **Local Search**: Hill climbing and simulated annealing for fine-tuning

### Evaluation Utilities
- Sparsity and compression ratio metrics
- Accuracy preservation evaluation
- Hardware performance estimation (speedup, memory, energy)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demonstration script:

```bash
python examples/basic_pruning_demo.py
```

This will show:
- Magnitude-based pruning
- Structured pruning for hardware efficiency
- Genetic algorithm optimization
- Local search optimization

## Usage Example

```python
import numpy as np
from src.pruning import MagnitudePruner
from src.algorithms import GeneticAlgorithm
from src.utils.metrics import evaluate_pruning_quality

# Create a model (dictionary of weight matrices)
model = {
    'layer1': np.random.randn(100, 100),
    'layer2': np.random.randn(100, 100)
}

# Apply magnitude-based pruning
pruner = MagnitudePruner(target_sparsity=0.5)
pruned_model = pruner.prune_model(model)

# Evaluate compression
metrics = evaluate_pruning_quality(
    model, pruned_model,
    original_accuracy=0.90,
    pruned_accuracy=0.87
)

print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
print(f"Accuracy retention: {metrics['accuracy_retention']:.2%}")
```

## Project Structure

```
.
├── src/
│   ├── pruning/           # Pruning methods
│   │   ├── base_pruner.py
│   │   ├── magnitude_pruner.py
│   │   └── structured_pruner.py
│   ├── algorithms/        # AI optimization algorithms
│   │   ├── genetic_algorithm.py
│   │   ├── reinforcement_learning.py
│   │   └── local_search.py
│   └── utils/            # Utility functions
│       ├── metrics.py
│       └── hardware.py
├── tests/                # Unit tests
├── examples/             # Example scripts
└── requirements.txt
```

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

## Key Challenges Addressed

### 1. Accuracy Preservation
- Intelligent importance scoring for parameters
- Gradual pruning strategies
- AI-guided hyperparameter selection

### 2. Hardware-Friendly Compression
- Structured pruning (removes entire neurons/channels)
- Memory reduction estimation
- Inference speedup calculation

### 3. Hyperparameter Optimization
- Genetic algorithms for global search
- Reinforcement learning for adaptive strategies
- Local search for fine-tuning

## Contributing

This is an academic project focused on exploring AI algorithms for LLM model compression.
