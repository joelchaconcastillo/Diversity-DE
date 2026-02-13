"""
GPU-accelerated Differential Evolution with Diversity Promotion

This implementation uses PyTorch to leverage GPU acceleration and implements
diversity promotion through the best non-penalized approach, where diversity
is explicitly maintained throughout the evolutionary process.
"""

import torch
from typing import Callable, Optional, Tuple, Union
import warnings


class DiversityDE:
    """
    Differential Evolution optimizer with explicit diversity promotion.
    
    This implementation promotes diversity by maintaining non-penalized solutions
    and uses PyTorch for GPU acceleration.
    
    Parameters
    ----------
    objective_func : Callable
        The objective function to minimize. Should accept a torch.Tensor of shape
        (pop_size, dimensions) and return a torch.Tensor of shape (pop_size,)
    bounds : torch.Tensor
        Bounds for the search space, shape (dimensions, 2) where bounds[i, 0] 
        is the lower bound and bounds[i, 1] is the upper bound for dimension i
    pop_size : int, optional
        Population size (default: 50)
    F : float, optional
        Differential weight (mutation factor), range [0, 2] (default: 0.8)
    CR : float, optional
        Crossover probability, range [0, 1] (default: 0.9)
    diversity_weight : float, optional
        Weight for diversity promotion, range [0, 1] (default: 0.3)
    device : str, optional
        Device to use ('cuda', 'cpu', or 'mps'), default: 'cuda' if available
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        objective_func: Callable[[torch.Tensor], torch.Tensor],
        bounds: torch.Tensor,
        pop_size: int = 50,
        F: float = 0.8,
        CR: float = 0.9,
        diversity_weight: float = 0.3,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.diversity_weight = diversity_weight
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Move bounds to device
        self.bounds = self.bounds.to(self.device)
        self.dimensions = self.bounds.shape[0]
        
        # Initialize population
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_solution = None
        self.best_fitness = None
        
        # Diversity metrics
        self.diversity_history = []
        
    def initialize_population(self) -> None:
        """Initialize the population randomly within bounds."""
        # Generate random population
        rand_vals = torch.rand(self.pop_size, self.dimensions, device=self.device)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.population = lower + rand_vals * (upper - lower)
        
        # Evaluate initial population
        self.fitness = self.objective_func(self.population)
        self.best_idx = torch.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].clone()
        self.best_fitness = self.fitness[self.best_idx].clone()
        
    def compute_diversity(self) -> float:
        """
        Compute population diversity as average pairwise distance.
        
        Returns
        -------
        float
            Average Euclidean distance between all pairs of solutions
        """
        # Compute pairwise distances
        diff = self.population.unsqueeze(1) - self.population.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        
        # Average over upper triangle (excluding diagonal)
        n = self.pop_size
        diversity = distances.sum() / (n * (n - 1))
        return diversity.item()
    
    def mutate(self) -> torch.Tensor:
        """
        Perform mutation using DE/best/1 strategy with diversity promotion.
        
        For each individual, creates a mutant vector using:
        v = best + F * (r1 - r2) + diversity_weight * (r3 - r4)
        
        where r1, r2, r3, r4 are random individuals from the population.
        
        Returns
        -------
        torch.Tensor
            Mutant vectors, shape (pop_size, dimensions)
        """
        # Generate random indices for mutation
        indices = torch.arange(self.pop_size, device=self.device)
        
        # For each individual, select 4 random different individuals
        mutants = torch.zeros_like(self.population)
        
        for i in range(self.pop_size):
            # Get available indices (excluding current individual)
            available = indices[indices != i]
            
            # Randomly select 4 different individuals
            selected = available[torch.randperm(len(available), device=self.device)[:4]]
            r1, r2, r3, r4 = selected
            
            # DE/best/1 mutation with diversity promotion
            # v = best + F * (r1 - r2) + diversity_weight * (r3 - r4)
            mutants[i] = (
                self.population[self.best_idx] + 
                self.F * (self.population[r1] - self.population[r2]) +
                self.diversity_weight * (self.population[r3] - self.population[r4])
            )
        
        return mutants
    
    def crossover(self, mutants: torch.Tensor) -> torch.Tensor:
        """
        Perform binomial crossover between population and mutants.
        
        Parameters
        ----------
        mutants : torch.Tensor
            Mutant vectors, shape (pop_size, dimensions)
            
        Returns
        -------
        torch.Tensor
            Trial vectors, shape (pop_size, dimensions)
        """
        # Generate random crossover mask
        cross_mask = torch.rand(self.pop_size, self.dimensions, device=self.device) < self.CR
        
        # Ensure at least one dimension is taken from mutant
        rand_dim = torch.randint(0, self.dimensions, (self.pop_size,), device=self.device)
        for i in range(self.pop_size):
            cross_mask[i, rand_dim[i]] = True
        
        # Create trial vectors
        trials = torch.where(cross_mask, mutants, self.population)
        
        return trials
    
    def bound_constrain(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Constrain vectors to be within bounds using reflection.
        
        Parameters
        ----------
        vectors : torch.Tensor
            Vectors to constrain, shape (pop_size, dimensions)
            
        Returns
        -------
        torch.Tensor
            Constrained vectors
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        
        # Clip to bounds
        vectors = torch.clamp(vectors, lower, upper)
        
        return vectors
    
    def select(self, trials: torch.Tensor) -> None:
        """
        Perform selection between current population and trial vectors.
        
        Parameters
        ----------
        trials : torch.Tensor
            Trial vectors, shape (pop_size, dimensions)
        """
        # Evaluate trial vectors
        trial_fitness = self.objective_func(trials)
        
        # Select better solutions (greedy selection)
        improved = trial_fitness < self.fitness
        
        self.population = torch.where(
            improved.unsqueeze(1),
            trials,
            self.population
        )
        self.fitness = torch.where(improved, trial_fitness, self.fitness)
        
        # Update best solution
        current_best_idx = torch.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_idx = current_best_idx
            self.best_solution = self.population[self.best_idx].clone()
            self.best_fitness = self.fitness[self.best_idx].clone()
    
    def step(self) -> Tuple[torch.Tensor, float]:
        """
        Perform one generation of the evolutionary process.
        
        Returns
        -------
        Tuple[torch.Tensor, float]
            Best solution and its fitness value
        """
        # Mutation
        mutants = self.mutate()
        
        # Bound constraint
        mutants = self.bound_constrain(mutants)
        
        # Crossover
        trials = self.crossover(mutants)
        
        # Bound constraint again
        trials = self.bound_constrain(trials)
        
        # Selection
        self.select(trials)
        
        # Track diversity
        diversity = self.compute_diversity()
        self.diversity_history.append(diversity)
        
        return self.best_solution.clone(), self.best_fitness.item()
    
    def optimize(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = True,
        min_iterations: int = 10,
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        Run the optimization process.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of generations (default: 1000)
        tolerance : float, optional
            Convergence tolerance (default: 1e-6)
        verbose : bool, optional
            Whether to print progress (default: True)
        min_iterations : int, optional
            Minimum number of iterations before checking convergence (default: 10)
            
        Returns
        -------
        Tuple[torch.Tensor, float, dict]
            Best solution, best fitness, and optimization history
        """
        # Initialize population if not already done
        if self.population is None:
            self.initialize_population()
        
        # History tracking
        history = {
            'best_fitness': [],
            'diversity': [],
            'mean_fitness': [],
        }
        
        if verbose:
            print(f"Starting optimization on {self.device}")
            print(f"Population size: {self.pop_size}, Dimensions: {self.dimensions}")
            print(f"F: {self.F}, CR: {self.CR}, Diversity weight: {self.diversity_weight}")
            print("-" * 70)
        
        # Main optimization loop
        for iteration in range(max_iterations):
            best_sol, best_fit = self.step()
            
            # Record history
            history['best_fitness'].append(best_fit)
            history['diversity'].append(self.diversity_history[-1])
            history['mean_fitness'].append(self.fitness.mean().item())
            
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1:4d}: "
                      f"Best = {best_fit:.6e}, "
                      f"Mean = {self.fitness.mean():.6e}, "
                      f"Diversity = {self.diversity_history[-1]:.6e}")
            
            # Check convergence (only after minimum iterations)
            if iteration >= min_iterations and abs(history['best_fitness'][-1] - history['best_fitness'][-2]) < tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
        
        if verbose:
            print("-" * 70)
            print(f"Optimization complete!")
            print(f"Best fitness: {self.best_fitness.item():.6e}")
            print(f"Final diversity: {self.diversity_history[-1]:.6e}")
        
        return self.best_solution.cpu(), self.best_fitness.item(), history
