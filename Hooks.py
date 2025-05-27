import torch
import logging
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradientClippingHook:
    """Hook to clip gradients during training to prevent explosions."""
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def __call__(self, module, grad_input, grad_output):
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    torch.nn.utils.clip_grad_norm_(g, self.max_norm)
                    logger.debug(f"Gradient clipped to max norm {self.max_norm}")

class LossSpikeDetectionHook:
    """Hook to detect and log loss spikes during training."""
    def __init__(self, threshold=10.0, window=5):
        self.threshold = threshold
        self.window = window
        self.losses = []

    def __call__(self, module, input, output):
        if isinstance(module, nn.Module) and output is not None:
            loss = output.mean() if output.requires_grad else None
            if loss is not None:
                self.losses.append(loss.item())
                if len(self.losses) > self.window:
                    self.losses.pop(0)
                    avg_loss = sum(self.losses[:-1]) / (len(self.losses) - 1)
                    current_loss = self.losses[-1]
                    if current_loss > avg_loss * self.threshold:
                        logger.warning(f"Loss spike detected: {current_loss:.4f} > {avg_loss:.4f} * {self.threshold}")

# Utility function to register hooks on a model
def register_azure_hooks(model):
    grad_clip_hook = GradientClippingHook(max_norm=1.0)
    loss_spike_hook = LossSpikeDetectionHook(threshold=10.0, window=5)
    for module in model.modules():
        module.register_full_backward_hook(grad_clip_hook)
        module.register_forward_hook(loss_spike_hook)
    logger.info("Azure hooks registered: GradientClippingHook and LossSpikeDetectionHook")