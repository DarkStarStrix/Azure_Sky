"""
AzureSky Optimizer implementation.
"""
import torch
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Azure(optim.Optimizer):
    """
    AzureSky Optimizer: A hybrid optimizer combining Adam with Simulated Annealing.
    
    This optimizer uses Simulated Annealing (SA) during early steps for exploration,
    gradually transitioning to Adam-based exploitation via dynamic temperature scaling
    and a sigmoid fusion schedule.
    """

    def __init__(self, params, lr=1e-3, T0=1.0, sigma=0.1, betas=(0.9, 0.999), eps=1e-8,
                 sa_steps=1000, sa_momentum=0.9, clip_grad_norm=1.0):
        """
        Initialize the Azure optimizer.
        
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            T0 (float): Initial temperature for Simulated Annealing.
            sigma (float): Noise scaling factor.
            betas (tuple): Coefficients for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            sa_steps (int): Number of steps during which Simulated Annealing is active.
            sa_momentum (float): Momentum factor for Simulated Annealing.
            clip_grad_norm (float, optional): Max norm of the gradients for clipping.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        # Ensure parameters are correctly formatted for PyTorch's optim.Optimizer
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = []
            for group in params:
                group_dict = group.copy()
                if 'params' not in group_dict:
                    raise ValueError("Each parameter group must contain a 'params' key")
                if isinstance(group_dict['params'], (list, tuple)) and len(group_dict['params']) > 0 and isinstance(group_dict['params'][0], tuple):
                    group_dict['params'] = [p for _, p in group_dict['params']]
                param_groups.append(group_dict)
            params = param_groups
        else:
            if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], tuple):
                params = [p for _, p in params]
            params = [{'params': params}]

        defaults = dict(lr=lr, T0=T0, sigma=sigma, betas=betas, eps=eps, sa_steps=sa_steps,
                        sa_momentum=sa_momentum, clip_grad_norm=clip_grad_norm)
        super().__init__(params, defaults)
        
        self.step_count = 0
        self.sa_active = True
        self.losses = []
        self.loss_window = 5
        self.loss_spike_threshold = 10.0

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

        if loss is not None:
            self._monitor_loss(loss.item())

        for group in self.param_groups:
            if group['clip_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], group['clip_grad_norm'])

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                T = self._compute_temperature(group)
                alpha = self._compute_alpha(group)

                if self.sa_active:
                    noise = torch.randn_like(p.data) * group['sigma'] * T
                    sa_update = noise
                else:
                    sa_update = torch.zeros_like(p.data)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                lr = group.get('lr', self.defaults['lr'])
                adam_update = -lr * m_hat / (v_hat.sqrt() + group['eps'])

                update = alpha * adam_update + (1 - alpha) * sa_update
                p.data.add_(update)

        self.step_count += 1
        if self.step_count >= self.param_groups[0]['sa_steps']:
            self.sa_active = False

        return loss

    def _compute_temperature(self, group):
        """
        Dynamic Temperature Scaling based on step progress.
        """
        epoch_decay = 0.05
        return group['T0'] * (1.0 / (1.0 + epoch_decay * self.step_count))

    def _compute_alpha(self, group):
        """
        Exploration-Exploitation Fusion Schedule using sigmoid.
        """
        midpoint = group['sa_steps'] / 2.0
        # Prevent overflow in exp
        val = -(self.step_count - midpoint) / (midpoint / 5.0)
        val = max(min(val, 100), -100)
        return 1.0 / (1.0 + np.exp(val))

    def _monitor_loss(self, loss):
        """
        Monitors for loss spikes and logs warnings.
        """
        self.losses.append(loss)
        if len(self.losses) > self.loss_window:
            self.losses.pop(0)
            avg_loss = sum(self.losses[:-1]) / (len(self.losses) - 1)
            current_loss = self.losses[-1]
            if avg_loss > 0 and current_loss > avg_loss * self.loss_spike_threshold:
                logger.warning(f"Loss spike detected: {current_loss:.4f} > {avg_loss:.4f} * {self.loss_spike_threshold}")
