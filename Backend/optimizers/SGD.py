"""
Stochastic Gradient Descent (SGD) Optimizer implementation.
"""
import torch
from .base import BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer implementation.
    
    This optimizer updates parameters using the gradient of the loss function,
    optionally with momentum.
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        """
        Initialize the SGD optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            momentum (float): Momentum factor.
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.state = {p: {'velocity': torch.zeros_like(p.data)} for p in self.params}

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            
        Returns:
            float: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for p in self.params:
            if p.grad is None:
                continue
                
            state = self.state[p]
            
            # Update velocity
            state['velocity'] = self.momentum * state['velocity'] - self.lr * p.grad
            
            # Update parameters
            p.data += state['velocity']
            
        return loss

    def zero_grad(self):
        """
        Clear the gradients of all optimized parameters.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self):
        """
        Return the state of the optimizer as a dictionary.
        
        Returns:
            dict: The optimizer state.
        """
        return {id(p): {'velocity': state['velocity'].clone()} for p, state in self.state.items()}

    def load_state_dict(self, state_dict):
        """
        Load the optimizer state from a dictionary.
        
        Args:
            state_dict (dict): The optimizer state.
        """
        param_map = {id(p): p for p in self.params}
        for pid, state in state_dict.items():
            if pid in param_map:
                p = param_map[pid]
                self.state[p] = {'velocity': state['velocity'].clone()}

    def __repr__(self):
        return f"SGDOptimizer(lr={self.lr}, momentum={self.momentum})"
