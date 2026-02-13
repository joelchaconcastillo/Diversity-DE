"""
Visualization example showing diversity preservation during optimization.

This example demonstrates how diversity is maintained throughout the
optimization process and creates plots comparing:
- Fitness convergence
- Diversity evolution
- Comparison with/without diversity promotion
"""

import torch
import matplotlib.pyplot as plt
from diversity_de import DiversityDE, rastrigin
from diversity_de.benchmark_functions import get_bounds


def run_optimization_with_tracking(diversity_weight, seed=42):
    """Run optimization and return history."""
    dimensions = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = get_bounds('rastrigin', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=rastrigin,
        bounds=bounds,
        pop_size=50,
        F=0.8,
        CR=0.9,
        diversity_weight=diversity_weight,
        device=device,
        seed=seed,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=500,
        tolerance=1e-10,
        verbose=False,
    )
    
    return best_fitness, history


def main():
    print("Diversity-DE Visualization Example")
    print("=" * 70)
    print("Optimizing Rastrigin function (10D) with different diversity weights")
    print("-" * 70)
    
    # Run optimizations with different diversity weights
    weights = [0.0, 0.2, 0.4, 0.6]
    results = {}
    
    for weight in weights:
        print(f"Running with diversity_weight = {weight}...", end=' ')
        fitness, history = run_optimization_with_tracking(weight)
        results[weight] = (fitness, history)
        print(f"Best fitness: {fitness:.6e}")
    
    print("-" * 70)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diversity-DE Performance Analysis on Rastrigin Function', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Fitness convergence
    ax1 = axes[0, 0]
    for weight in weights:
        _, history = results[weight]
        ax1.semilogy(history['best_fitness'], 
                    label=f'diversity_weight={weight}',
                    linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness (log scale)')
    ax1.set_title('Fitness Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diversity evolution
    ax2 = axes[0, 1]
    for weight in weights:
        _, history = results[weight]
        ax2.plot(history['diversity'], 
                label=f'diversity_weight={weight}',
                linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Diversity')
    ax2.set_title('Diversity Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean fitness
    ax3 = axes[1, 0]
    for weight in weights:
        _, history = results[weight]
        ax3.semilogy(history['mean_fitness'], 
                    label=f'diversity_weight={weight}',
                    linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Mean Population Fitness (log scale)')
    ax3.set_title('Mean Fitness Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final results comparison
    ax4 = axes[1, 1]
    final_fitness = [results[w][0] for w in weights]
    final_diversity = [results[w][1]['diversity'][-1] for w in weights]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([f'{w}' for w in weights], final_fitness, 
                    alpha=0.7, color='steelblue', label='Best Fitness')
    bars2 = ax4_twin.bar([f'{w}' for w in weights], final_diversity,
                         alpha=0.5, color='orange', label='Final Diversity')
    
    ax4.set_xlabel('Diversity Weight')
    ax4.set_ylabel('Best Fitness', color='steelblue')
    ax4_twin.set_ylabel('Final Diversity', color='orange')
    ax4.set_title('Final Performance vs Diversity Trade-off')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'diversity_de_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Display summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'Div Weight':<12} {'Best Fitness':<20} {'Final Diversity':<20} {'Avg Diversity'}")
    print("-" * 70)
    
    for weight in weights:
        fitness, history = results[weight]
        avg_div = sum(history['diversity']) / len(history['diversity'])
        print(f"{weight:<12.1f} {fitness:<20.6e} {history['diversity'][-1]:<20.6e} {avg_div:.6e}")
    
    print("=" * 70)
    print("\nKey Insights:")
    print("- Higher diversity weights maintain more exploration")
    print("- Trade-off between exploitation (low diversity) and exploration (high diversity)")
    print("- GPU acceleration enables efficient population-based search")
    
    # Show plot (comment out if running headless)
    # plt.show()


if __name__ == "__main__":
    main()
