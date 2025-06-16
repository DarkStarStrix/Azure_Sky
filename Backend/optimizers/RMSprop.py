# RmsProp optimizer implementation
from abc import ABC

from optimizers.base import BaseOptimizer

class RMSpropOptimizer(BaseOptimizer, ABC):
    """
    RMSprop optimizer implementation.
    This optimizer uses a moving average of squared gradients to normalize the gradient.
    """

    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.state = {p: {'mean_square': 0} for p in params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            state = self.state[p]
            state['mean_square'] = self.alpha * state['mean_square'] + (1 - self.alpha) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (state['mean_square'].sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def __repr__(self):
        return f"RMSpropOptimizer(lr={self.lr}, alpha={self.alpha}, eps={self.eps})"

    def state_dict(self):
        return {p: {'mean_square': state['mean_square']} for p, state in self.state.items()}
