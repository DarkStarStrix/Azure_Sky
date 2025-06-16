# Brent function benchmark

import numpy as np
import scipy.optimize as opt
from Base import BaseBenchmark

class Brent(BaseBenchmark):
    """Brent's function benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "Brent"
        self.dimensions = 1
        self.global_minimum = 1.0
        self.global_minimum_value = 0.0

    @staticmethod
    def evaluate(x):
        """Evaluate Brent's function."""
        return (x - 1)**2 * (x + 1)**2 * (x - 2)**2

def brent_function(x):
    """Brent's function."""
    return (x - 1)**2 * (x + 1)**2 * (x - 2)**2

def benchmark_brent():
    """Benchmark the Brent function."""
    np.random.uniform (-2, 2)
    result = opt.minimize_scalar(brent_function, bounds=(-2, 2), method='bounded')

    print(f"Optimized parameter: {result.x}")
    print(f"Function value at optimum: {result.fun}")
    print("Optimization successful:", result.success)

if __name__ == "__main__":
    benchmark_brent()
