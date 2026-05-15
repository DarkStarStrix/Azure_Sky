"""
Ackley N 2 function benchmark.
"""
import torch
import numpy as np
from .Base import BaseBenchmark

class AckleyN2(BaseBenchmark):
    """
    Ackley N 2 function benchmark.
    
    A widely used multimodal test function for optimization algorithms.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Ackley N 2"
        self.dimensions = 10
        self.set_global_min([0.0] * self.dimensions)
        self.global_minimum_value = 0.0

    @staticmethod
    def evaluate(x):
        """
        Evaluate the Ackley N 2 function using PyTorch operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N,).
            
        Returns:
            torch.Tensor: Evaluated value.
        """
        a = 20.0
        b = 0.2
        c = 2.0 * torch.pi
        n = x.size(0)
        
        sum1 = torch.sum(x ** 2)
        sum2 = torch.sum(torch.cos(c * x))
        
        term1 = -a * torch.exp(-b * torch.sqrt(sum1 / n + 1e-12))
        term2 = -torch.exp(sum2 / n)
        
        return term1 + term2 + a + torch.exp(torch.tensor(1.0))
