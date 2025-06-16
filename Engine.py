from Backend.Benchmarks import Himmelblau, Adjiman , Brent , Ackley 
from Backend.optimizers import adam as Adam, SGD as SGD, azure_optim as AzureSky , RMSprop
from Backend.ML_Tasks import MNISTRunner, CIFAR10Runner
import numpy as np
from Metrics import calculate_benchmark_metrics, calculate_ml_metrics
from Plots import plot_benchmark_surface, plot_ml_curves

class Engine:
    def __init__(self):
        self.benchmarks = {
            'Himmelblau': Himmelblau,
            'Adjiman': Adjiman,
            'Brent': Brent,
            'Ackley': Ackley
        }
        self.optimizers = {
            'Adam': Adam,
            'SGD': SGD,
            'AzureSky': AzureSky,
            'RMSprop': RMSprop
        }
        self.ml_tasks = {
            'MNIST': MNISTRunner,
            'CIFAR10': CIFAR10Runner
        }

    def run_benchmark(self, benchmark_name, optimizer_name, initial_guess):
        benchmark = self.benchmarks [benchmark_name]
        optimizer = self.optimizers [optimizer_name]
        result = optimizer.optimize(benchmark, initial_guess)
        metrics = calculate_benchmark_metrics(result['final_point'], benchmark.global_min, result['path'], result['loss_values'])
        plot_benchmark_surface(benchmark, result['path'])
        return metrics

    def run_ml_task(self, task_name, epochs=10):
        task = self.ml_tasks [task_name]
        history = task.run(epochs)
        metrics = calculate_ml_metrics(history['train'], history['val'])
        plot_ml_curves(history)
        return metrics

    def list_benchmarks(self):
        return list(self.benchmarks.keys())

    def list_optimizers(self):
        return list(self.optimizers.keys())

    def list_ml_tasks(self):
        return list(self.ml_tasks.keys())

    def run(self):
        # This method can be used to run the engine in a specific mode or with specific parameters.
        # For now, it just returns the list of benchmarks and optimizers.
        return {
            'benchmarks': self.list_benchmarks(),
            'optimizers': self.list_optimizers(),
            'ml_tasks': self.list_ml_tasks()
        }


if __name__ == "__main__":
    engine = Engine()

    # Example usage for benchmarks
    benchmark_name = 'Himmelblau'
    optimizer_name = 'Adam'
    initial_guess = np.random.uniform(-5, 5, size=2)
    benchmark_metrics = engine.run_benchmark(benchmark_name, optimizer_name, initial_guess)
    print(f"Benchmark Metrics for {benchmark_name} with {optimizer_name}: {benchmark_metrics}")

    # Example usage for ML tasks
    ml_task_name = 'MNIST'
    ml_metrics = engine.run_ml_task(ml_task_name, epochs=5)
    print(f"ML Task Metrics for {ml_task_name}: {ml_metrics}")
