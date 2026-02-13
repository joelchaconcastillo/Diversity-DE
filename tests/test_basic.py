"""
Basic tests for Diversity-DE implementation.

These tests verify the core functionality of the optimizer.
"""

import torch
from diversity_de import DiversityDE, sphere, rastrigin
from diversity_de.benchmark_functions import get_bounds


def test_basic_optimization():
    """Test basic optimization on sphere function."""
    dimensions = 5
    device = 'cpu'
    bounds = get_bounds('sphere', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=sphere,
        bounds=bounds,
        pop_size=30,
        device=device,
        seed=42,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=100,
        verbose=False,
        min_iterations=10,
    )
    
    # Check that optimization improved
    assert len(history['best_fitness']) > 0
    assert best_fitness < 100, "Should find reasonable solution"
    
    # Check that solution is within bounds
    assert torch.all(best_solution >= bounds[:, 0])
    assert torch.all(best_solution <= bounds[:, 1])
    
    print("✓ Basic optimization test passed")


def test_diversity_tracking():
    """Test that diversity is tracked correctly."""
    dimensions = 5
    device = 'cpu'
    bounds = get_bounds('sphere', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=sphere,
        bounds=bounds,
        pop_size=30,
        diversity_weight=0.3,
        device=device,
        seed=42,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=50,
        verbose=False,
        min_iterations=10,
    )
    
    # Check diversity is tracked
    assert len(history['diversity']) == len(history['best_fitness'])
    assert all(d > 0 for d in history['diversity'])
    
    print("✓ Diversity tracking test passed")


def test_device_handling():
    """Test that device selection works correctly."""
    dimensions = 5
    
    # Test CPU
    bounds_cpu = get_bounds('sphere', dimensions, device='cpu')
    optimizer_cpu = DiversityDE(
        objective_func=sphere,
        bounds=bounds_cpu,
        pop_size=20,
        device='cpu',
    )
    assert optimizer_cpu.device.type == 'cpu'
    
    # Test auto-selection
    optimizer_auto = DiversityDE(
        objective_func=sphere,
        bounds=bounds_cpu,
        pop_size=20,
    )
    assert optimizer_auto.device is not None
    
    print("✓ Device handling test passed")


def test_multimodal_function():
    """Test on multimodal function (Rastrigin)."""
    dimensions = 5
    device = 'cpu'
    bounds = get_bounds('rastrigin', dimensions, device=device)
    
    optimizer = DiversityDE(
        objective_func=rastrigin,
        bounds=bounds,
        pop_size=50,
        diversity_weight=0.4,
        device=device,
        seed=42,
    )
    
    best_solution, best_fitness, history = optimizer.optimize(
        max_iterations=100,
        verbose=False,
        min_iterations=20,
    )
    
    # Rastrigin is harder, just check it runs and improves
    assert len(history['best_fitness']) > 0
    assert best_fitness < 200, "Should make some progress"
    
    print("✓ Multimodal function test passed")


def test_reproducibility():
    """Test that using same seed gives same results."""
    dimensions = 5
    device = 'cpu'
    bounds = get_bounds('sphere', dimensions, device=device)
    
    # Run 1
    opt1 = DiversityDE(sphere, bounds, pop_size=20, device=device, seed=123)
    sol1, fit1, _ = opt1.optimize(max_iterations=50, verbose=False, min_iterations=10)
    
    # Run 2 with same seed
    opt2 = DiversityDE(sphere, bounds, pop_size=20, device=device, seed=123)
    sol2, fit2, _ = opt2.optimize(max_iterations=50, verbose=False, min_iterations=10)
    
    # Results should be identical
    assert torch.allclose(sol1, sol2, rtol=1e-5)
    assert abs(fit1 - fit2) < 1e-6
    
    print("✓ Reproducibility test passed")


if __name__ == "__main__":
    print("Running Diversity-DE tests...")
    print("=" * 60)
    
    test_basic_optimization()
    test_diversity_tracking()
    test_device_handling()
    test_multimodal_function()
    test_reproducibility()
    
    print("=" * 60)
    print("All tests passed! ✓")
