"""
Basic example of using Diversity-DE to optimize the Sphere function.

This example demonstrates:
- Setting up the optimizer with GPU acceleration
- Running optimization on a simple benchmark function
- Accessing and displaying results
"""

import torch
from diversity_de import DiversityDE, sphere
from diversity_de.benchmark_functions import get_bounds


def main():
    # Configuration
    dimensions = 10
    pop_size = 50
    max_iterations = 500
    
    # Set device (will use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get bounds for sphere function
    bounds = get_bounds('sphere', dimensions, device=device)
    
    # Create optimizer
    optimizer = DiversityDE(
        objective_func=sphere,
        bounds=bounds,
        pop_size=pop_size,
        F=0.8,                    # Differential weight
        CR=0.9,                   # Crossover probability
        diversity_weight=0.1,     # Lower diversity for unimodal function
        device=device,
        seed=42,                  # For reproducibility
    )
    
    # Run optimization
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=max_iterations,
        tolerance=1e-8,
        verbose=True,
        min_iterations=100,  # Ensure sufficient iterations
    )
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best solution found: {best_solution.numpy()}")
    print(f"Best fitness value: {best_fitness:.10e}")
    print(f"Expected optimal value: 0.0")
    print(f"Final population diversity: {history['diversity'][-1]:.6e}")
    print("="*70)
    
    # Check if solution is close to global optimum
    expected_optimum = torch.zeros(dimensions)
    distance_to_optimum = torch.norm(best_solution - expected_optimum).item()
    print(f"\nDistance to global optimum: {distance_to_optimum:.6e}")
    
    if best_fitness < 1e-6:
        print("✓ Successfully found the global optimum!")
    else:
        print("✗ Did not reach global optimum, but found a good solution.")


if __name__ == "__main__":
    main()
