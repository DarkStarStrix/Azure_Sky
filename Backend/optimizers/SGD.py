# SGD implementation

from base import BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer implementation.
    This optimizer updates parameters using the gradient of the loss function.
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.state = {p: {'velocity': 0} for p in params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            state = self.state[p]
            state['velocity'] = self.momentum * state['velocity'] - self.lr * p.grad
            p.data += state['velocity']

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def __repr__(self):
        return f"SGDOptimizer(lr={self.lr}, momentum={self.momentum})"

    def state_dict(self):
        return {p: {'velocity': state['velocity']} for p, state in self.state.items()}

    def load_state_dict(self, state_dict):
        for p in self.params:
            if p in state_dict:
                self.state[p] = state_dict[p]
            else:
                self.state[p] = {'velocity': 0}
    def __str__(self):
        return f"SGDOptimizer(lr={self.lr}, momentum={self.momentum})"
