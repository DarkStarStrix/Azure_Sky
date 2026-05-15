"""
Brent function benchmark.
"""
import torch
from .Base import BaseBenchmark

class Brent(BaseBenchmark):
    """
    Brent's function benchmark.
    
    A 1D test function with multiple local minima.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Brent"
        self.dimensions = 1
        # Brent function minimum is typically at x=1 or x=2 (global minimum value is 0)
        # We'll set the global minimum to 1.0 for metrics calculation
        self.set_global_min([1.0])
        self.global_minimum_value = 0.0

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Brent function using PyTorch operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (1,).
            
        Returns:
            torch.Tensor: Evaluated value.
        """
        x_val = x[0]
        return (x_val - 1)**2 * (x_val + 1)**2 * (x_val - 2)**2
