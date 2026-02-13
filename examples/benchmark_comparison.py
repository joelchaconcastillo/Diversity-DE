"""
Advanced example demonstrating optimization on multiple benchmark functions
with diversity tracking and comparison.

This example:
- Tests multiple benchmark functions
- Compares optimization with and without diversity promotion
- Tracks and reports diversity metrics
"""

import torch
from diversity_de import DiversityDE, sphere, rastrigin, rosenbrock, ackley, griewank
from diversity_de.benchmark_functions import get_bounds


def optimize_function(func, func_name, dimensions=10, with_diversity=True):
    """Run optimization on a single function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = get_bounds(func_name, dimensions, device=device)
    
    diversity_weight = 0.3 if with_diversity else 0.0
    
    optimizer = DiversityDE(
        objective_func=func,
        bounds=bounds,
        pop_size=50,
        F=0.8,
        CR=0.9,
        diversity_weight=diversity_weight,
        device=device,
        seed=42,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=500,
        tolerance=1e-8,
        verbose=False,
    )
    
    return best_fitness, history


def main():
    print("GPU-Accelerated Diversity-DE Benchmark Suite")
    print("=" * 80)
    
    # Test functions
    test_functions = [
        (sphere, 'sphere'),
        (rastrigin, 'rastrigin'),
        (rosenbrock, 'rosenbrock'),
        (ackley, 'ackley'),
        (griewank, 'griewank'),
    ]
    
    dimensions = 10
    
    print(f"\nTesting on {dimensions}-dimensional problems")
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 80)
    print(f"{'Function':<15} {'With Diversity':<20} {'Without Diversity':<20} {'Improvement'}")
    print("-" * 80)
    
    for func, func_name in test_functions:
        # Run with diversity promotion
        fitness_with, history_with = optimize_function(
            func, func_name, dimensions, with_diversity=True
        )
        
        # Run without diversity promotion
        fitness_without, history_without = optimize_function(
            func, func_name, dimensions, with_diversity=False
        )
        
        # Calculate improvement
        if fitness_without > 0:
            improvement = ((fitness_without - fitness_with) / fitness_without) * 100
        else:
            improvement = 0.0
        
        # Display results
        print(f"{func_name:<15} {fitness_with:<20.6e} {fitness_without:<20.6e} "
              f"{improvement:>6.2f}%")
    
    print("-" * 80)
    print("\nKey observations:")
    print("- Diversity promotion helps maintain exploration")
    print("- GPU acceleration enables faster convergence")
    print("- Performance varies by problem landscape")
    
    # Detailed example with one function
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: Rastrigin Function (highly multimodal)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = get_bounds('rastrigin', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=rastrigin,
        bounds=bounds,
        pop_size=100,
        F=0.8,
        CR=0.9,
        diversity_weight=0.4,  # Higher diversity weight for multimodal
        device=device,
        seed=42,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=1000,
        tolerance=1e-10,
        verbose=True,
    )
    
    print(f"\nBest fitness achieved: {best_fitness:.10e}")
    print(f"Global optimum: 0.0")
    print(f"Average diversity maintained: {sum(history['diversity'])/len(history['diversity']):.6e}")


if __name__ == "__main__":
    main()
