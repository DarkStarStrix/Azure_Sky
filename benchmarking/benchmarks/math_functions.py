"""
Scalable synthetic mathematical benchmark functions.

All functions are implemented in pure PyTorch, support arbitrary dimensionality,
have known global minima, and are differentiable everywhere (or near-everywhere)
so that gradient-based optimizers can be evaluated fairly.
"""
import torch
import numpy as np


class ScalableBenchmark:
    """
    Base class for scalable mathematical benchmark functions.
    """

    name: str = "BaseBenchmark"
    global_min_value: float = 0.0

    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimensionality of the search space.
        """
        self.dim = dim

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the benchmark function at x.

        Args:
            x (torch.Tensor): Parameter tensor of shape (dim,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        raise NotImplementedError

    def global_minimum(self) -> np.ndarray:
        """Return the known global minimum location as a NumPy array."""
        raise NotImplementedError

    def distance_to_global_min(self, x: torch.Tensor) -> float:
        """Compute Euclidean distance from x to the known global minimum."""
        gmin = torch.tensor(self.global_minimum(), dtype=x.dtype)
        return (x.detach().cpu() - gmin).norm().item()


class AckleyND(ScalableBenchmark):
    """
    N-dimensional Ackley function.

    Highly multimodal with a nearly flat outer region and a deep global minimum
    at the origin. Exponentially harder as dimensionality increases due to the
    proliferation of local minima. Global minimum: f(0,...,0) = 0.
    """

    name = "Ackley"
    global_min_value = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = 20.0, 0.2, 2.0 * torch.pi
        n = x.size(0)
        sum_sq = torch.sum(x ** 2)
        sum_cos = torch.sum(torch.cos(c * x))
        term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / n + 1e-12))
        term2 = -torch.exp(sum_cos / n)
        return term1 + term2 + a + torch.exp(torch.tensor(1.0))

    def global_minimum(self) -> np.ndarray:
        return np.zeros(self.dim)


class RastriginND(ScalableBenchmark):
    """
    N-dimensional Rastrigin function.

    Highly multimodal with a regular grid of local minima. The periodic cosine
    term creates a landscape that is extremely hostile to pure gradient descent.
    Global minimum: f(0,...,0) = 0.
    """

    name = "Rastrigin"
    global_min_value = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        A = 10.0
        return A * n + torch.sum(x ** 2 - A * torch.cos(2.0 * torch.pi * x))

    def global_minimum(self) -> np.ndarray:
        return np.zeros(self.dim)


class RosenbrockND(ScalableBenchmark):
    """
    N-dimensional Rosenbrock (banana) function.

    A classic test of an optimizer's ability to navigate a narrow, curved
    valley. The global minimum lies inside the valley but is hard to reach
    because the gradient along the valley floor is very small. Particularly
    challenging for momentum-based methods.
    Global minimum: f(1,...,1) = 0.
    """

    name = "Rosenbrock"
    global_min_value = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        xi  = x[:-1]
        xi1 = x[1:]
        return torch.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1.0 - xi) ** 2)

    def global_minimum(self) -> np.ndarray:
        return np.ones(self.dim)


class SchwefelND(ScalableBenchmark):
    """
    N-dimensional Schwefel function.

    Deceptive: the global minimum is geometrically distant from the next-best
    local minimum, meaning optimizers that converge too quickly will almost
    certainly miss it. A strong test of global exploration capability.
    Global minimum: f(420.97,...,420.97) ≈ 0.
    """

    name = "Schwefel"
    global_min_value = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        return 418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x) + 1e-12)))

    def global_minimum(self) -> np.ndarray:
        return np.full(self.dim, 420.9687)


class NonConvexDial(ScalableBenchmark):
    """
    Tunable non-convexity benchmark.

    Constructed as a base quadratic (perfectly convex) plus a sinusoidal
    perturbation term whose amplitude is controlled by `alpha`. This provides
    a continuous dial from trivially convex (alpha=0) to highly non-convex
    (large alpha), enabling controlled ablation of optimizer robustness to
    landscape ruggedness.

    f(x) = ||x||^2 + alpha * sum(sin(freq * x_i))

    Global minimum: approximately at origin for small alpha; shifts and
    multiplies as alpha increases.
    """

    name = "NonConvexDial"
    global_min_value = 0.0

    def __init__(self, dim: int, alpha: float = 1.0, freq: float = 5.0):
        """
        Args:
            dim (int): Dimensionality.
            alpha (float): Non-convexity amplitude. 0 = pure quadratic.
            freq (float): Frequency of the sinusoidal perturbation.
        """
        super().__init__(dim)
        self.alpha = alpha
        self.freq = freq
        self.name = f"NonConvexDial(alpha={alpha})"

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        quadratic = torch.sum(x ** 2)
        # Shift the sinusoidal perturbation so the function stays non-negative:
        # sin(freq*x) ranges in [-1,1], so adding dim*alpha ensures f(x) >= 0.
        perturbation = self.alpha * torch.sum(torch.sin(self.freq * x) + 1.0)
        return quadratic + perturbation

    def global_minimum(self) -> np.ndarray:
        return np.zeros(self.dim)


class HimmelblauND(ScalableBenchmark):
    """
    N-dimensional generalised Himmelblau function.

    The classic 2D Himmelblau has four global minima of equal value (f=0),
    making it an ideal test of an optimizer's ability to find *any* of several
    equally good solutions rather than converging to a single basin. The N-D
    generalisation pairs consecutive dimensions into (x_{2i}, x_{2i+1}) blocks,
    each contributing a standard Himmelblau term, and sums them. For odd
    dimensionality the final unpaired dimension contributes a simple quadratic.

    This structure preserves the multi-modal, deceptive character of the
    original at arbitrary scale. Global minimum value: 0.0 at any of the
    four known 2D minima replicated across all dimension pairs.

    Known 2D global minima (each pair):
        (3.0,       2.0      )  f = 0
        (-2.805118, 3.131312 )  f = 0
        (-3.779310, -3.283186)  f = 0
        ( 3.584428, -1.848126)  f = 0
    """

    name = "Himmelblau"
    global_min_value = 0.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        total = torch.tensor(0.0, dtype=x.dtype)
        # Process pairs of dimensions
        for i in range(0, n - 1, 2):
            xi  = x[i]
            xi1 = x[i + 1]
            total = total + (xi ** 2 + xi1 - 11.0) ** 2 + (xi + xi1 ** 2 - 7.0) ** 2
        # Unpaired final dimension (odd n)
        if n % 2 == 1:
            total = total + x[-1] ** 2
        return total

    def global_minimum(self) -> np.ndarray:
        """
        Return the canonical global minimum (3, 2) replicated across all pairs.
        """
        gmin = np.zeros(self.dim)
        for i in range(0, self.dim - 1, 2):
            gmin[i]     = 3.0
            gmin[i + 1] = 2.0
        return gmin


BENCHMARK_MAP = {
    'Ackley':     AckleyND,
    'Rastrigin':  RastriginND,
    'Rosenbrock': RosenbrockND,
    'Schwefel':   SchwefelND,
    'Himmelblau': HimmelblauND,
}
