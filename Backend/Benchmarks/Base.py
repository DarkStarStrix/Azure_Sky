"""
Base Benchmark class.
"""
import torch
import numpy as np

class BaseBenchmark:
    """
    Base class for all optimization benchmarks.
    
    Provides logic to define global variables and methods that can be used by
    all benchmarks, minimizing code duplication. Rewritten to support PyTorch
    tensors for seamless integration with PyTorch optimizers.
    """

    def __init__(self):
        self.global_min = None
        self.initial_guess = None
        self.path = []
        self.loss_values = []
        self.dimensions = 2
        self.name = "Base"

    def set_global_min(self, point):
        """Set the global minimum point."""
        self.global_min = np.array(point)

    def set_initial_guess(self, guess):
        """Set the initial guess for optimization."""
        self.initial_guess = guess

    def add_to_path(self, point):
        """Record a point in the optimization path."""
        if isinstance(point, torch.Tensor):
            self.path.append(point.detach().cpu().numpy().copy())
        else:
            self.path.append(np.array(point).copy())

    def add_loss_value(self, value):
        """Record a loss value."""
        if isinstance(value, torch.Tensor):
            self.loss_values.append(value.item())
        else:
            self.loss_values.append(float(value))

    def reset(self):
        """Reset the benchmark state."""
        self.path.clear()
        self.loss_values.clear()

    def get_metrics(self):
        """
        Calculate metrics based on the optimization run.
        
        Returns:
            dict: Distance to global min, final loss, and convergence rate.
        """
        if self.global_min is None or not self.path or not self.loss_values:
            raise ValueError("Metrics cannot be calculated. Ensure global_min, path, and loss_values are set.")
            
        distance = np.linalg.norm(self.path[-1] - self.global_min)
        convergence_rate = len(self.path) if self.loss_values[-1] < 1e-5 else float('inf')
        
        return {
            'distance': float(distance),
            'final_loss': float(self.loss_values[-1]),
            'convergence_rate': convergence_rate
        }

    @staticmethod
    def evaluate(x):
        """
        Evaluate the benchmark function. Must be implemented by subclasses.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Evaluated value.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")
        
    def f(self, x):
        """
        Wrapper for numpy/scipy evaluation (used in plotting).
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            float: Evaluated value.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return self.evaluate(x_tensor).item()
