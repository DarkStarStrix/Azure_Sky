"""
RMSprop Optimizer implementation.
"""
import torch
from .base import BaseOptimizer

class RMSpropOptimizer(BaseOptimizer):
    """
    RMSprop optimizer implementation.
    
    This optimizer uses a moving average of squared gradients to normalize the gradient.
    """

    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        """
        Initialize the RMSprop optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            alpha (float): Smoothing constant.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.state = {p: {'mean_square': torch.zeros_like(p.data)} for p in self.params}

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
            
            # Update moving average of squared gradients
            state['mean_square'] = self.alpha * state['mean_square'] + (1 - self.alpha) * (p.grad ** 2)
            
            # Update parameters
            p.data -= self.lr * p.grad / (state['mean_square'].sqrt() + self.eps)
            
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
        return {id(p): {'mean_square': state['mean_square'].clone()} for p, state in self.state.items()}

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
                self.state[p] = {'mean_square': state['mean_square'].clone()}

    def __repr__(self):
        return f"RMSpropOptimizer(lr={self.lr}, alpha={self.alpha}, eps={self.eps})"
