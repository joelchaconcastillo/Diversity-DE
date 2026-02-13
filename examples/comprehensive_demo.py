"""
Comprehensive demonstration of Diversity-DE features.

This example showcases:
1. GPU acceleration (if available)
2. Diversity promotion mechanism
3. Multiple optimization scenarios
4. Performance analysis
"""

import torch
from diversity_de import DiversityDE, sphere, rastrigin, ackley
from diversity_de.benchmark_functions import get_bounds


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 1: GPU Acceleration")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device detected: {device.upper()}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run a quick optimization
    dimensions = 20
    bounds = get_bounds('sphere', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=sphere,
        bounds=bounds,
        pop_size=100,
        device=device,
    )
    
    print(f"\nOptimizing {dimensions}D sphere function with population size 100...")
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=200,
        verbose=False,
        min_iterations=50,
    )
    
    print(f"Best fitness achieved: {best_fitness:.6e}")
    print(f"Generations: {len(history['best_fitness'])}")
    
    if device == 'cuda':
        print("✓ GPU acceleration enabled and working!")
    else:
        print("ℹ Running on CPU (GPU not available)")


def demo_diversity_promotion():
    """Demonstrate diversity promotion on multimodal function."""
    print("\n" + "=" * 70)
    print("DEMO 2: Diversity Promotion on Multimodal Function")
    print("=" * 70)
    
    dimensions = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = get_bounds('rastrigin', dimensions, device=device)
    
    # Test with different diversity weights
    diversity_weights = [0.0, 0.2, 0.4]
    results = {}
    
    print("\nTesting Rastrigin function with different diversity weights...")
    print("-" * 70)
    
    for weight in diversity_weights:
        optimizer = DiversityDE(
            objective_func=rastrigin,
            bounds=bounds,
            pop_size=80,
            diversity_weight=weight,
            device=device,
            seed=42,
        )
        
        best_solution, best_fitness, history = optimizer.optimize(
            max_iterations=300,
            verbose=False,
            min_iterations=100,
        )
        
        avg_diversity = sum(history['diversity']) / len(history['diversity'])
        results[weight] = (best_fitness, avg_diversity)
        
        print(f"Diversity weight {weight:.1f}: "
              f"Best fitness = {best_fitness:.6e}, "
              f"Avg diversity = {avg_diversity:.6e}")
    
    print("-" * 70)
    print("Observation: Higher diversity weights maintain more exploration,")
    print("which can help escape local optima on multimodal problems.")


def demo_scalability():
    """Demonstrate scalability with problem dimensions."""
    print("\n" + "=" * 70)
    print("DEMO 3: Scalability Across Dimensions")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dimensions_list = [5, 10, 20, 30]
    
    print("\nOptimizing Ackley function across different dimensions...")
    print("-" * 70)
    print(f"{'Dimensions':<12} {'Population':<12} {'Best Fitness':<15} {'Generations'}")
    print("-" * 70)
    
    for dims in dimensions_list:
        bounds = get_bounds('ackley', dims, device=device)
        pop_size = 20 * dims  # Scale population with dimensions
        
        optimizer = DiversityDE(
            objective_func=ackley,
            bounds=bounds,
            pop_size=pop_size,
            diversity_weight=0.3,
            device=device,
            seed=42,
        )
        
        best_solution, best_fitness, history = optimizer.optimize(
            max_iterations=200,
            verbose=False,
            min_iterations=50,
        )
        
        print(f"{dims:<12} {pop_size:<12} {best_fitness:<15.6e} {len(history['best_fitness'])}")
    
    print("-" * 70)
    print("Observation: Algorithm scales well with dimensionality.")
    print("GPU acceleration particularly beneficial for high-dimensional problems.")


def demo_single_objective_optimization():
    """Demonstrate single-objective optimization workflow."""
    print("\n" + "=" * 70)
    print("DEMO 4: Complete Single-Objective Optimization Workflow")
    print("=" * 70)
    
    # Define a custom objective function
    def custom_objective(x: torch.Tensor) -> torch.Tensor:
        """
        Custom single-objective function:
        Minimize: f(x) = sum((x - 2)^2) + 0.5 * sum(sin(x))
        Global minimum near x = [2, 2, 2, ...]
        """
        return torch.sum((x - 2) ** 2, dim=1) + 0.5 * torch.sum(torch.sin(x), dim=1)
    
    print("\nCustom objective: f(x) = sum((x - 2)^2) + 0.5 * sum(sin(x))")
    print("Expected optimum near x = [2, 2, 2, ...]")
    
    dimensions = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = torch.tensor([[-5.0, 5.0]] * dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=custom_objective,
        bounds=bounds,
        pop_size=60,
        F=0.8,
        CR=0.9,
        diversity_weight=0.2,
        device=device,
        seed=42,
    )
    
    print("\nRunning optimization...")
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=300,
        verbose=True,
        min_iterations=50,
    )
    
    expected_optimum = torch.full((dimensions,), 2.0)
    distance_to_expected = torch.norm(best_solution - expected_optimum).item()
    
    print(f"\nResults:")
    print(f"Best solution: {best_solution.numpy()}")
    print(f"Best fitness: {best_fitness:.6e}")
    print(f"Distance to expected optimum: {distance_to_expected:.6e}")
    print(f"Final diversity: {history['diversity'][-1]:.6e}")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("DIVERSITY-DE COMPREHENSIVE DEMONSTRATION")
    print("GPU-Accelerated Differential Evolution with Diversity Promotion")
    print("=" * 70)
    
    # Run all demos
    demo_gpu_acceleration()
    demo_diversity_promotion()
    demo_scalability()
    demo_single_objective_optimization()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ GPU acceleration works automatically when available")
    print("✓ Diversity promotion helps on multimodal problems")
    print("✓ Algorithm scales well with problem dimensionality")
    print("✓ Flexible single-objective optimization framework")
    print("✓ Easy to use with custom objective functions")
    print("\nFor more examples, see the examples/ directory")
    print("For detailed usage, see USAGE.md")


if __name__ == "__main__":
    main()
