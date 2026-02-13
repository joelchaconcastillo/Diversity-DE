"""
Benchmark functions for testing optimization algorithms.

All functions are implemented to work with PyTorch tensors and support
batch evaluation for GPU acceleration.
"""

import torch
import math


def sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Sphere function: f(x) = sum(x_i^2)
    
    Global minimum: f(0, ..., 0) = 0
    Search domain: typically [-5.12, 5.12]^n
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (pop_size, dimensions)
        
    Returns
    -------
    torch.Tensor
        Function values of shape (pop_size,)
    """
    return torch.sum(x ** 2, dim=1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Global minimum: f(0, ..., 0) = 0
    Search domain: typically [-5.12, 5.12]^n
    Highly multimodal with many local minima
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (pop_size, dimensions)
        
    Returns
    -------
    torch.Tensor
        Function values of shape (pop_size,)
    """
    n = x.shape[1]
    return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * math.pi * x), dim=1)


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Global minimum: f(1, ..., 1) = 0
    Search domain: typically [-5, 10]^n
    Valley-shaped, difficult to optimize
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (pop_size, dimensions)
        
    Returns
    -------
    torch.Tensor
        Function values of shape (pop_size,)
    """
    x_current = x[:, :-1]
    x_next = x[:, 1:]
    return torch.sum(100 * (x_next - x_current ** 2) ** 2 + (1 - x_current) ** 2, dim=1)


def ackley(x: torch.Tensor) -> torch.Tensor:
    """
    Ackley function: 
    f(x) = -20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e
    
    Global minimum: f(0, ..., 0) = 0
    Search domain: typically [-32.768, 32.768]^n
    Highly multimodal with a nearly flat outer region
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (pop_size, dimensions)
        
    Returns
    -------
    torch.Tensor
        Function values of shape (pop_size,)
    """
    n = x.shape[1]
    sum_sq = torch.sum(x ** 2, dim=1)
    sum_cos = torch.sum(torch.cos(2 * math.pi * x), dim=1)
    
    term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / n))
    term2 = -torch.exp(sum_cos / n)
    
    return term1 + term2 + 20 + math.e


def griewank(x: torch.Tensor) -> torch.Tensor:
    """
    Griewank function:
    f(x) = 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i+1)))
    
    Global minimum: f(0, ..., 0) = 0
    Search domain: typically [-600, 600]^n
    Multimodal with many widespread local minima
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (pop_size, dimensions)
        
    Returns
    -------
    torch.Tensor
        Function values of shape (pop_size,)
    """
    n = x.shape[1]
    sum_sq = torch.sum(x ** 2, dim=1) / 4000
    
    # Create indices for denominator
    indices = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    cos_term = torch.cos(x / torch.sqrt(indices))
    prod_cos = torch.prod(cos_term, dim=1)
    
    return 1 + sum_sq - prod_cos


def get_bounds(function_name: str, dimensions: int, device: str = 'cpu') -> torch.Tensor:
    """
    Get recommended bounds for a benchmark function.
    
    Parameters
    ----------
    function_name : str
        Name of the benchmark function
    dimensions : int
        Number of dimensions
    device : str, optional
        Device to create tensor on (default: 'cpu')
        
    Returns
    -------
    torch.Tensor
        Bounds tensor of shape (dimensions, 2)
    """
    bounds_dict = {
        'sphere': (-5.12, 5.12),
        'rastrigin': (-5.12, 5.12),
        'rosenbrock': (-5.0, 10.0),
        'ackley': (-32.768, 32.768),
        'griewank': (-600.0, 600.0),
    }
    
    if function_name.lower() not in bounds_dict:
        raise ValueError(f"Unknown function: {function_name}. "
                        f"Available: {list(bounds_dict.keys())}")
    
    lower, upper = bounds_dict[function_name.lower()]
    bounds = torch.tensor([[lower, upper]] * dimensions, dtype=torch.float32, device=device)
    
    return bounds
