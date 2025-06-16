# Himmelblau function benchmark

from time import time
from Base import BaseBenchmark
from numpy.random import default_rng
from scipy.optimize import minimize

class Himmelblau(BaseBenchmark):
    """Himmelblau's function benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "Himmelblau"
        self.dimensions = 2
        self.global_minimum = [3, 2]
        self.global_minimum_value = 0

    @staticmethod
    def evaluate(x):
        """Evaluate the Himmelblau function."""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def himmelblau(x):
    """Himmelblau's function."""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def benchmark_himmelblau():
    """Benchmark the Himmelblau function."""
    rng = default_rng()
    x0 = rng.uniform(-5, 5, size=2)
    start_time = time()
    result = minimize(himmelblau, x0, method='BFGS')
    end_time = time()

    print(f"Optimized parameters: {result.x}")
    print(f"Function value at optimum: {result.fun}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
