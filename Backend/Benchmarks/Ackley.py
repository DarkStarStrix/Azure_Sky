# Ackley N 2 function Benchmark

import numpy as np
from scipy.optimize import minimize
from Base import BaseBenchmark

class AckleyN2(BaseBenchmark):
    """Ackley N 2 function benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "Ackley N 2"
        self.dimensions = 10
        self.global_minimum = [0] * self.dimensions
        self.global_minimum_value = 0.0

    @staticmethod
    def evaluate(x):
        """Evaluate the Ackley N 2 function."""
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)

        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(np.cos(c * xi) for xi in x)

        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)

        return term1 + term2 + a + np.exp(1)

def ackley_n2(x):
    """Ackley N 2 function."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + a + np.exp(1)

def benchmark_ackley_n2():
    """Benchmark the Ackley N 2 function."""
    x0 = np.random.uniform(-5, 5, size=10)
    result = minimize(ackley_n2, x0, method='BFGS')

    print(f"Optimized parameters: {result.x}")
    print(f"Function value at optimum: {result.fun}")
    print("Optimization successful:", result.success)

if __name__ == "__main__":
    benchmark_ackley_n2()