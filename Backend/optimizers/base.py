# The base class for all optimizers. Acts as an interface for the optimizers.

from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Base class for all optimizers.
    """

    @abstractmethod
    def step(self):
        """
        Perform a single optimization step.
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
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Load the optimizer state from a dictionary.
        """
        pass
