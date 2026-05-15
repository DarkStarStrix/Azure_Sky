"""
Himmelblau function benchmark.
"""
import torch
from .Base import BaseBenchmark

class Himmelblau(BaseBenchmark):
    """
    Himmelblau's function benchmark.
    
    A 2D multi-modal function, used to test the performance of optimization algorithms.
    It has four identical local minima.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Himmelblau"
        self.dimensions = 2
        # One of the global minima
        self.set_global_min([3.0, 2.0])
        self.global_minimum_value = 0.0

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Himmelblau function using PyTorch operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (2,).
            
        Returns:
            torch.Tensor: Evaluated value.
        """
        x1, x2 = x[0], x[1]
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
