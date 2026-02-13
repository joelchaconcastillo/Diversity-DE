# Diversity-DE: GPU-Accelerated Differential Evolution with Diversity Promotion

A PyTorch-based implementation of Differential Evolution (DE) with explicit diversity promotion for single-objective optimization. This implementation leverages GPU acceleration for high-performance optimization while maintaining population diversity throughout the evolutionary process.

## Features

- 🚀 **GPU Acceleration**: Full PyTorch implementation for CUDA/MPS acceleration
- 🎯 **Diversity Promotion**: Explicit diversity preservation using best non-penalized approach
- 🔧 **Single-Objective Optimization**: Focused on efficient single-objective optimization
- 📊 **Benchmark Functions**: Built-in suite of standard test functions
- 🎨 **Easy to Use**: Simple API with sensible defaults
- 📦 **Modern Python**: Uses `uv` for fast, reliable dependency management

## Installation

This project uses `uv` for dependency management. First, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install the package:

```bash
git clone https://github.com/joelchaconcastillo/Diversity-DE.git
cd Diversity-DE
uv pip install -e .
```

For development with visualization tools:

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from diversity_de import DiversityDE, sphere
from diversity_de.benchmark_functions import get_bounds

# Setup
dimensions = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bounds = get_bounds('sphere', dimensions, device=device)

# Create optimizer
optimizer = DiversityDE(
    objective_func=sphere,
    bounds=bounds,
    pop_size=50,
    F=0.8,                    # Differential weight
    CR=0.9,                   # Crossover probability
    diversity_weight=0.3,     # Diversity promotion weight
    device=device,
)

# Optimize
best_solution, best_fitness, history = optimizer.optimize(
    max_iterations=500,
    verbose=True,
)

print(f"Best fitness: {best_fitness}")
print(f"Best solution: {best_solution}")
```

## Usage with `uv`

Run examples directly with `uv`:

```bash
# Basic optimization example
uv run examples/basic_example.py

# Benchmark comparison
uv run examples/benchmark_comparison.py

# Visualization (requires matplotlib)
uv run examples/visualization_example.py
```

## Algorithm Overview

### Differential Evolution with Diversity Promotion

This implementation uses the DE/best/1 mutation strategy with an additional diversity promotion term:

```
v_i = x_best + F * (x_r1 - x_r2) + diversity_weight * (x_r3 - x_r4)
```

Where:
- `x_best`: Current best solution
- `F`: Differential weight (controls mutation magnitude)
- `x_r1, x_r2, x_r3, x_r4`: Randomly selected individuals
- `diversity_weight`: Weight for diversity promotion term

### Key Features

1. **Best Non-Penalized Approach**: Uses the best solution found so far without penalties, maintaining a clear objective
2. **Explicit Diversity**: Additional mutation term promotes population diversity
3. **GPU Acceleration**: Parallel evaluation of entire population on GPU
4. **Adaptive Selection**: Greedy selection ensures monotonic improvement

## Parameters

### DiversityDE Constructor

- `objective_func`: Callable that takes a batch of solutions and returns fitness values
- `bounds`: Search space bounds (dimensions × 2 tensor)
- `pop_size`: Population size (default: 50)
- `F`: Differential weight, range [0, 2] (default: 0.8)
- `CR`: Crossover probability, range [0, 1] (default: 0.9)
- `diversity_weight`: Diversity promotion weight, range [0, 1] (default: 0.3)
- `device`: Device for computation ('cuda', 'cpu', or 'mps')
- `seed`: Random seed for reproducibility

### optimize() Method

- `max_iterations`: Maximum number of generations (default: 1000)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `verbose`: Print progress during optimization (default: True)

## Benchmark Functions

Built-in benchmark functions for testing:

- **Sphere**: Simple unimodal function
- **Rastrigin**: Highly multimodal with many local minima
- **Rosenbrock**: Valley-shaped, difficult to optimize
- **Ackley**: Multimodal with nearly flat outer region
- **Griewank**: Multimodal with widespread local minima

Each function supports batch evaluation on GPU.

## Examples

### Basic Usage

See `examples/basic_example.py` for a simple example optimizing the Sphere function.

### Benchmark Comparison

See `examples/benchmark_comparison.py` to compare performance with and without diversity promotion across multiple benchmark functions.

### Visualization

See `examples/visualization_example.py` to generate plots showing:
- Fitness convergence
- Diversity evolution
- Performance trade-offs

## Performance

GPU acceleration provides significant speedup for population-based optimization:

- **CPU**: ~10-50 generations/second (depends on problem)
- **GPU (CUDA)**: ~100-500 generations/second (10-50× faster)

The exact speedup depends on:
- Population size
- Problem dimensionality
- GPU hardware
- Function evaluation complexity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{diversity_de,
  title = {Diversity-DE: GPU-Accelerated Differential Evolution with Diversity Promotion},
  author = {Chacon Castillo, Joel},
  year = {2026},
  url = {https://github.com/joelchaconcastillo/Diversity-DE}
}
```

## References

- Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11(4), 341-359.
- Price, K., Storn, R. M., & Lampinen, J. A. (2006). Differential evolution: a practical approach to global optimization. Springer Science & Business Media.