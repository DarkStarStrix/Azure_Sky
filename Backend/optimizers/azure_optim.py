import torch
import torch.optim as optim
import numpy as np
import logging

# Configure logging for loss monitoring
logging.basicConfig (level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger (__name__)


class Azure (optim.Optimizer):
    def __init__(self, params, lr=0.0007518383921113902, T0=2.2723218904585964, sigma=0.17181058166567398,
                 betas=(0.9, 0.999), eps=1e-8, sa_steps=5, sa_momentum=0.6612913488540948, clip_grad_norm=1.0):
        """
        Azure Sky Optimizer: A hybrid optimizer combining Simulated Annealing (SA) and Adam.

        Args:
            params (iterable): Iterable of parameters or dicts defining parameter groups.
            lr (float): Learning rate for Adam phase (default: 0.0007518383921113902).
            T0 (float): Initial temperature for SA (default: 2.2723218904585964).
            sigma (float): Perturbation strength for SA (default: 0.17181058166567398).
            betas (tuple): Adam's exponential decay rates (default: (0.9, 0.999)).
            eps (float): Adam's epsilon for numerical stability (default: 1e-8).
            sa_steps (int): Number of steps for SA phase (default: 5).
            sa_momentum (float): Momentum for SA updates (default: 0.6612913488540948).
            clip_grad_norm (float): Max norm for gradient clipping (default: 1.0).
        """
        # Process params to handle various input formats
        if isinstance (params, (list, tuple)) and isinstance (params [0], dict):
            # Handle parameter groups (e.g., [{'params': ..., 'lr': ...}, ...])
            param_groups = []
            for group in params:
                group_dict = group.copy ()
                if 'params' not in group_dict:
                    raise ValueError ("Each parameter group must contain a 'params' key")
                # Convert named_parameters() to a list of parameters if necessary
                if isinstance (group_dict ['params'], (list, tuple)) and isinstance (group_dict ['params'] [0], tuple):
                    group_dict ['params'] = [p for _, p in group_dict ['params']]
                param_groups.append (group_dict)
            params = param_groups
        else:
            # Handle direct parameter lists or named_parameters()
            if isinstance (params, (list, tuple)) and isinstance (params [0], tuple):
                params = [p for _, p in params]  # Convert named_parameters() to parameter list
            params = [{'params': params}]

        # Set defaults for each parameter group
        defaults = dict (lr=lr, T0=T0, sigma=sigma, betas=betas, eps=eps, sa_steps=sa_steps,
                         sa_momentum=sa_momentum, clip_grad_norm=clip_grad_norm)
        super ().__init__ (params, defaults)
        self.step_count = 0
        self.sa_active = True
        self.losses = []
        self.loss_window = 5
        self.loss_spike_threshold = 10.0

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad ():
                loss = closure ()

        # Loss spike monitoring
        if loss is not None:
            self._monitor_loss (loss.item ())

        for group in self.param_groups:
            # Gradient clipping
            if group ['clip_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_ (group ['params'], group ['clip_grad_norm'])

            for p in group ['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Dynamic Temperature Scaling
                T = self._compute_temperature (group)
                # Exploration-Exploitation Fusion
                alpha = self._compute_alpha (group)

                if self.sa_active:
                    noise = torch.randn_like (p.data) * group ['sigma'] * T
                    sa_update = noise
                else:
                    sa_update = torch.zeros_like (p.data)

                # Adam update
                state = self.state [p]
                if 'm' not in state:
                    state ['m'] = torch.zeros_like (p.data)
                    state ['v'] = torch.zeros_like (p.data)
                    state ['step'] = 0
                m, v = state ['m'], state ['v']
                beta1, beta2 = group ['betas']
                state ['step'] += 1
                m.mul_ (beta1).add_ (grad, alpha=1 - beta1)
                v.mul_ (beta2).addcmul_ (grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** state ['step'])
                v_hat = v / (1 - beta2 ** state ['step'])
                # Use group-specific learning rate if provided
                lr = group.get ('lr', self.defaults ['lr'])
                adam_update = -lr * m_hat / (v_hat.sqrt () + group ['eps'])

                # Combined update
                update = alpha * adam_update + (1 - alpha) * sa_update
                p.data.add_ (update)

        self.step_count += 1
        if self.step_count >= self.param_groups [0] ['sa_steps']:
            self.sa_active = False
        return loss

    def _compute_temperature(self, group):
        """Dynamic Temperature Scaling based on step progress."""
        epoch_decay = 0.05  # Adjustable decay rate
        return group ['T0'] * (1.0 / (1.0 + epoch_decay * self.step_count))

    def _compute_alpha(self, group):
        """Exploration-Exploitation Fusion Schedule using sigmoid."""
        midpoint = group ['sa_steps'] / 2
        return 1 / (1 + np.exp (-(self.step_count - midpoint) / (midpoint / 5)))

    def _monitor_loss(self, loss):
        """Monitors for loss spikes and logs warnings."""
        self.losses.append (loss)
        if len (self.losses) > self.loss_window:
            self.losses.pop (0)
            avg_loss = sum (self.losses [:-1]) / (len (self.losses) - 1)
            current_loss = self.losses [-1]
            if current_loss > avg_loss * self.loss_spike_threshold:
                logger.warning (
                    f"Loss spike detected: {current_loss:.4f} > {avg_loss:.4f} * {self.loss_spike_threshold}")
