# Usage Guide

## Quick Start with `uv`

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/joelchaconcastillo/Diversity-DE.git
cd Diversity-DE

# Install dependencies
uv pip install -e .
```

### Running Examples

```bash
# Basic optimization example
uv run examples/basic_example.py

# Benchmark comparison across multiple functions
uv run examples/benchmark_comparison.py

# Visualization example (requires matplotlib)
uv run examples/visualization_example.py
```

## Basic Usage

### Simple Optimization

```python
import torch
from diversity_de import DiversityDE, sphere
from diversity_de.benchmark_functions import get_bounds

# Define problem
dimensions = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bounds = get_bounds('sphere', dimensions, device=device)

# Create optimizer
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
    pop_size=50,
    diversity_weight=0.3,
    device=device,
)

# Run optimization
best_solution, best_fitness, history = optimizer.optimize(
    max_iterations=500,
    verbose=True,
)
```

### Custom Objective Function

```python
import torch
from diversity_de import DiversityDE

# Define custom objective function
def my_objective(x: torch.Tensor) -> torch.Tensor:
    """
    Custom objective function.
    
    Args:
        x: Input tensor of shape (pop_size, dimensions)
        
    Returns:
        Fitness values of shape (pop_size,)
    """
    # Example: minimize sum of squares with penalty
    return torch.sum(x ** 2, dim=1) + torch.sum(torch.abs(x), dim=1)

# Define bounds
dimensions = 5
bounds = torch.tensor([[-10.0, 10.0]] * dimensions)

# Optimize
optimizer = DiversityDE(
    objective_func=my_objective,
    bounds=bounds,
    device='cuda',  # Use GPU
)

best_solution, best_fitness, history = optimizer.optimize(
    max_iterations=300,
)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Parameter Tuning

### For Unimodal Problems (e.g., Sphere)

```python
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
    pop_size=50,
    F=0.8,
    CR=0.9,
    diversity_weight=0.1,  # Lower diversity for exploitation
)
```

### For Multimodal Problems (e.g., Rastrigin)

```python
optimizer = DiversityDE(
    objective_func=rastrigin,
    bounds=bounds,
    pop_size=100,           # Larger population
    F=0.8,
    CR=0.9,
    diversity_weight=0.4,   # Higher diversity for exploration
)
```

### For High-Dimensional Problems

```python
optimizer = DiversityDE(
    objective_func=your_function,
    bounds=bounds,
    pop_size=200,           # Even larger population
    F=0.8,
    CR=0.9,
    diversity_weight=0.3,
)
```

## GPU Acceleration

The optimizer automatically uses GPU if available:

```python
# Automatic device selection
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
)

# Force CPU
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
    device='cpu',
)

# Force CUDA
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
    device='cuda',
)
```

## Accessing Optimization History

```python
best_solution, best_fitness, history = optimizer.optimize(max_iterations=500)

# Access history
print(f"Best fitness per generation: {history['best_fitness']}")
print(f"Diversity per generation: {history['diversity']}")
print(f"Mean fitness per generation: {history['mean_fitness']}")

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness Convergence')
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.plot(history['diversity'])
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.title('Diversity Evolution')

plt.subplot(1, 3, 3)
plt.plot(history['mean_fitness'])
plt.xlabel('Generation')
plt.ylabel('Mean Fitness')
plt.title('Mean Fitness')
plt.yscale('log')

plt.tight_layout()
plt.savefig('optimization_history.png')
```

## Available Benchmark Functions

```python
from diversity_de import (
    sphere,      # f(0,...,0) = 0, unimodal
    rastrigin,   # f(0,...,0) = 0, highly multimodal
    rosenbrock,  # f(1,...,1) = 0, valley-shaped
    ackley,      # f(0,...,0) = 0, multimodal
    griewank,    # f(0,...,0) = 0, multimodal
)
from diversity_de.benchmark_functions import get_bounds

# Get recommended bounds for any function
bounds = get_bounds('rastrigin', dimensions=10, device='cuda')
```

## Running Tests

```bash
# Run all tests
python tests/test_basic.py

# Or with uv
uv run tests/test_basic.py
```

## Troubleshooting

### GPU Out of Memory

If you encounter GPU memory issues:

```python
# Reduce population size
optimizer = DiversityDE(..., pop_size=30)

# Or use CPU
optimizer = DiversityDE(..., device='cpu')
```

### Slow Convergence

If optimization is slow to converge:

```python
# Increase population size
optimizer = DiversityDE(..., pop_size=100)

# Adjust diversity weight (lower for more exploitation)
optimizer = DiversityDE(..., diversity_weight=0.1)

# Increase F parameter
optimizer = DiversityDE(..., F=0.9)
```

### Poor Solution Quality on Multimodal Problems

```python
# Increase diversity weight
optimizer = DiversityDE(..., diversity_weight=0.5)

# Increase population size
optimizer = DiversityDE(..., pop_size=150)

# Run for more iterations
best_solution, best_fitness, history = optimizer.optimize(
    max_iterations=1000,
    min_iterations=200,  # Ensure enough exploration
)
```
