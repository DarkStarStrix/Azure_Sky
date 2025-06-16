# Adjiman function benchmark

import numpy as np
from scipy.optimize import minimize
from Base import BaseBenchmark

class Adjiman(BaseBenchmark):
    """Adjiman's function benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "Adjiman"
        self.dimensions = 2
        self.global_minimum = [0, 0]
        self.global_minimum_value = 0.5

    @staticmethod
    def evaluate(x):
        """Evaluate Adjiman's function."""
        x1, x2 = x
        term1 = (x1**2 + x2**2)**0.5
        term2 = np.sin(term1)
        term3 = np.exp(-term1)
        return 0.5 * (term1 + term2 + term3)

def adjiman(x):
    """Adjiman's function."""
    x1, x2 = x
    term1 = (x1**2 + x2**2)**0.5
    term2 = np.sin(term1)
    term3 = np.exp(-term1)
    return 0.5 * (term1 + term2 + term3)

def benchmark_adjiman():
    """Benchmark the Adjiman function."""
    x0 = np.random.uniform(-5, 5, size=2)
    result = minimize(adjiman, x0, method='BFGS')

    print(f"Optimized parameters: {result.x}")
    print(f"Function value at optimum: {result.fun}")
    print("Optimization successful:", result.success)

if __name__ == "__main__":
    benchmark_adjiman()