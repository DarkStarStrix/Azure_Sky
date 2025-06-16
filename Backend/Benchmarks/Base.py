# This file acts as the base class for benchmarks in the Backend/Benchmarks directory. it has logic to define global variables and methods that can be used by all benchmarks. and minimizes code duplication.
import numpy as np

class BaseBenchmark:
    def __init__(self):
        self.global_min = None
        self.initial_guess = None
        self.path = []
        self.loss_values = []

    def set_global_min(self, point):
        self.global_min = point

    def set_initial_guess(self, guess):
        self.initial_guess = guess

    def add_to_path(self, point):
        self.path.append(point)

    def add_loss_value(self, value):
        self.loss_values.append(value)

    def reset(self):
        self.global_min = None
        self.initial_guess = None
        self.path.clear()
        self.loss_values.clear()

    def get_metrics(self):
        if self.global_min is None or not self.path or not self.loss_values:
            raise ValueError("Metrics cannot be calculated. Ensure global_min, path, and loss_values are set.")

        distance = np.linalg.norm(self.path[-1] - self.global_min)
        convergence_rate = len(self.path) if self.loss_values[-1] < 1e-5 else float('inf')
        return {
            'distance': float(distance),
            'final_loss': float(self.loss_values[-1]),
            'convergence_rate': convergence_rate
        }

    def __str__(self):
        return f"BaseBenchmark(global_min={self.global_min}, initial_guess={self.initial_guess}, path_length={len(self.path)}, loss_values_length={len(self.loss_values)})"

    def __repr__(self):
        return f"BaseBenchmark(global_min={self.global_min}, initial_guess={self.initial_guess}, path_length={len(self.path)}, loss_values_length={len(self.loss_values)})"

    def __eq__(self, other):
        if not isinstance(other, BaseBenchmark):
            return False
        return (self.global_min == other.global_min and
                self.initial_guess == other.initial_guess and
                self.path == other.path and
                self.loss_values == other.loss_values)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.global_min, tuple(self.initial_guess), tuple(self.path), tuple(self.loss_values)))

    def __len__(self):
        return len(self.path)
