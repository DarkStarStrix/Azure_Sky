# Adam optimizer implementation

from base import BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer implementation.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.state = {p: {'m': 0, 'v': 0, 't': 0} for p in params}

    def step(self):
        for p in self.params:
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

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def state_dict(self):
        return {p: {'m': state['m'], 'v': state['v'], 't': state['t']} for p, state in self.state.items()}

    def load_state_dict(self, state_dict):
        for p in self.params:
            if p in state_dict:
                self.state[p] = state_dict[p]
    def __repr__(self):
        return f"AdamOptimizer(lr={self.lr}, betas={self.betas}, eps={self.eps})"
