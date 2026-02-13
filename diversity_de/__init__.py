"""
Diversity-DE: GPU-accelerated Differential Evolution with Diversity Promotion
"""

from .optimizer import DiversityDE
from .benchmark_functions import (
    sphere,
    rastrigin,
    rosenbrock,
    ackley,
    griewank,
)

__version__ = "0.1.0"
__all__ = [
    "DiversityDE",
    "sphere",
    "rastrigin", 
    "rosenbrock",
    "ackley",
    "griewank",
]
