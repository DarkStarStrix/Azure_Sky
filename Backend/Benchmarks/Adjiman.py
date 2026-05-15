"""
Adjiman function benchmark.
"""
import torch
from .Base import BaseBenchmark

class Adjiman(BaseBenchmark):
    """
    Adjiman's function benchmark.
    
    A 2D continuous, non-convex optimization test function.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Adjiman"
        self.dimensions = 2
        self.set_global_min([0.0, 0.0])
        self.global_minimum_value = 0.5

    @staticmethod
    def evaluate(x):
        """
        Evaluate Adjiman's function using PyTorch operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (2,).
            
        Returns:
            torch.Tensor: Evaluated value.
        """
        x1, x2 = x[0], x[1]
        term1 = torch.sqrt(x1**2 + x2**2)
        term2 = torch.sin(term1)
        term3 = torch.exp(-term1)
        return 0.5 * (term1 + term2 + term3)
