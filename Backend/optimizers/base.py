"""
Base optimizer class.
"""
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Base class for all custom optimizers.
    
    Provides the required interface for optimization steps, gradient clearing,
    and state management, designed to work seamlessly with PyTorch parameters.
    """

    @abstractmethod
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        Returns:
            float: The loss value if closure is provided, else None.
        """
        pass

    @abstractmethod
    def zero_grad(self):
        """
        Clear the gradients of all optimized parameters.
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Return the state of the optimizer as a dictionary.
        
        Returns:
            dict: The optimizer state.
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Load the optimizer state from a dictionary.
        
        Args:
            state_dict (dict): The optimizer state.
        """
        pass
