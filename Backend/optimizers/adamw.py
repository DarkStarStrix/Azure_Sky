# AdamW implementation

from optimizers.adam import AdamOptimizer

class AdamWOptimizer(AdamOptimizer):
    """
    AdamW optimizer implementation.
    This optimizer decouples weight decay from the optimization steps.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
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

            # Update parameters with weight decay
            p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)

    def __repr__(self):
        return f"AdamWOptimizer(lr={self.lr}, betas={self.betas}, eps={self.eps}, weight_decay={self.weight_decay})"
