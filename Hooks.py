"""
PyTorch Hooks for monitoring and controlling gradients.
"""
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class GradientClippingHook:
    """
    A backward hook that clips gradients to a maximum norm.
    """
    def __init__(self, max_norm=1.0):
        """
        Initialize the hook.
        
        Args:
            max_norm (float): Maximum norm of the gradients.
        """
        self.max_norm = max_norm

    def __call__(self, module, grad_input, grad_output):
        """
        Apply gradient clipping.
        """
        if grad_input is not None:
            clipped_grad_input = []
            for grad in grad_input:
                if grad is not None:
                    norm = grad.norm(2)
                    if norm > self.max_norm:
                        clipped_grad = grad * (self.max_norm / norm)
                        clipped_grad_input.append(clipped_grad)
                    else:
                        clipped_grad_input.append(grad)
                else:
                    clipped_grad_input.append(None)
            return tuple(clipped_grad_input)
        return grad_input

class LossSpikeDetectionHook:
    """
    A forward hook that monitors the loss and detects sudden spikes.
    """
    def __init__(self, threshold=10.0, window=5):
        """
        Initialize the hook.
        
        Args:
            threshold (float): Multiplier above average loss to consider a spike.
            window (int): Moving average window size.
        """
        self.threshold = threshold
        self.window = window
        self.losses = []

    def __call__(self, module, input, output):
        """
        Monitor output (assumed to be loss) for spikes.
        """
        # Note: In PyTorch, loss is usually calculated after the forward pass,
        # so this hook is a bit unconventional unless attached to a specific loss module.
        if isinstance(output, torch.Tensor) and output.numel() == 1:
            current_loss = output.item()
            self.losses.append(current_loss)
            if len(self.losses) > self.window:
                self.losses.pop(0)
                avg_loss = sum(self.losses[:-1]) / (len(self.losses) - 1)
                if avg_loss > 0 and current_loss > avg_loss * self.threshold:
                    logger.warning(f"Loss spike detected: {current_loss:.4f} > {avg_loss:.4f} * {self.threshold}")

def register_azure_hooks(model):
    """
    Utility function to register Azure hooks on a model.
    
    Args:
        model (nn.Module): The model to attach hooks to.
    """
    grad_clip_hook = GradientClippingHook(max_norm=1.0)
    loss_spike_hook = LossSpikeDetectionHook(threshold=10.0, window=5)
    
    for module in model.modules():
        module.register_full_backward_hook(grad_clip_hook)
        module.register_forward_hook(loss_spike_hook)
        
    logger.info("Azure hooks registered: GradientClippingHook and LossSpikeDetectionHook")
