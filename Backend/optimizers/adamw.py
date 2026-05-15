"""
AdamW Optimizer implementation.
"""
import torch
from .adam import AdamOptimizer

class AdamWOptimizer(AdamOptimizer):
    """
    AdamW optimizer implementation.
    
    This optimizer decouples weight decay from the optimization steps.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize the AdamW optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay coefficient.
        """
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

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
            
            # Update parameters with decoupled weight decay
            p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)
            
        return loss

    def __repr__(self):
        return f"AdamWOptimizer(lr={self.lr}, betas={self.betas}, eps={self.eps}, weight_decay={self.weight_decay})"
