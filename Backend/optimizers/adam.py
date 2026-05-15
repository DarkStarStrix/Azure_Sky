"""
Adam Optimizer implementation.
"""
import torch
from .base import BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer implementation.
    
    This optimizer uses adaptive learning rates based on first and second moments
    of the gradients.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Initialize the Adam optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.state = {p: {'m': torch.zeros_like(p.data), 'v': torch.zeros_like(p.data), 't': 0} for p in self.params}

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
            state['t'] += 1
            
            # Update biased first moment estimate
            state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * p.grad
            
            # Update biased second raw moment estimate
            state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * (p.grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = state['m'] / (1 - self.betas[0] ** state['t'])
            
            # Compute bias-corrected second raw moment estimate
            v_hat = state['v'] / (1 - self.betas[1] ** state['t'])
            
            # Update parameters
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
            
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
        return {id(p): {'m': state['m'].clone(), 'v': state['v'].clone(), 't': state['t']} for p, state in self.state.items()}

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
                self.state[p] = {'m': state['m'].clone(), 'v': state['v'].clone(), 't': state['t']}

    def __repr__(self):
        return f"AdamOptimizer(lr={self.lr}, betas={self.betas}, eps={self.eps})"
