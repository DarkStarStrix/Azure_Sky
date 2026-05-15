"""
Engine module for orchestrating benchmarks and ML tasks.
"""
import torch
import numpy as np
from Backend.Benchmarks import Himmelblau, Adjiman, Brent, AckleyN2
from Backend.optimizers import AdamOptimizer, SGDOptimizer, Azure, RMSpropOptimizer
from Backend.ML_Tasks import MNISTRunner, CIFAR10Runner
from Metrics import calculate_benchmark_metrics, calculate_ml_metrics
from Plots import plot_benchmark_surface, plot_ml_curves

class Engine:
    """
    Orchestrates the execution of benchmarks and ML tasks using various optimizers.
    """
    
    def __init__(self):
        self.benchmarks = {
            'Himmelblau': Himmelblau,
            'Adjiman': Adjiman,
            'Brent': Brent,
            'Ackley': AckleyN2
        }
        self.optimizers = {
            'Adam': AdamOptimizer,
            'SGD': SGDOptimizer,
            'AzureSky': Azure,
            'RMSprop': RMSpropOptimizer
        }
        self.ml_tasks = {
            'MNIST': MNISTRunner,
            'CIFAR10': CIFAR10Runner
        }

    def run_benchmark(self, benchmark_name, optimizer_name, initial_guess, max_steps=1000, tolerance=1e-5):
        """
        Run a benchmark optimization task.
        
        Args:
            benchmark_name (str): Name of the benchmark function.
            optimizer_name (str): Name of the optimizer to use.
            initial_guess (list or np.ndarray): Initial starting point.
            max_steps (int): Maximum number of optimization steps.
            tolerance (float): Convergence tolerance.
            
        Returns:
            dict: Metrics from the optimization run.
        """
        benchmark_class = self.benchmarks[benchmark_name]
        benchmark = benchmark_class()
        benchmark.set_initial_guess(initial_guess)
        
        # Convert initial guess to a PyTorch parameter
        x = torch.tensor(initial_guess, dtype=torch.float32, requires_grad=True)
        
        optimizer_class = self.optimizers[optimizer_name]
        optimizer = optimizer_class([x], lr=0.1)
        
        benchmark.add_to_path(x)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            loss = benchmark.evaluate(x)
            loss.backward()
            optimizer.step()
            
            benchmark.add_to_path(x)
            benchmark.add_loss_value(loss)
            
            if loss.item() < tolerance:
                break
                
        metrics = benchmark.get_metrics()
        
        # Optional: Generate and save plot
        # fig = plot_benchmark_surface(benchmark, [np.array(benchmark.path)], [optimizer_name])
        # fig.savefig(f"{benchmark_name}_{optimizer_name}_surface.png")
        
        return metrics

    def run_ml_task(self, task_name, optimizer_name, epochs=10, batch_size=64, lr=0.001):
        """
        Run an ML classification task.
        
        Args:
            task_name (str): Name of the dataset/task.
            optimizer_name (str): Name of the optimizer to use.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            lr (float): Learning rate.
            
        Returns:
            dict: Metrics from the training run.
        """
        task_class = self.ml_tasks[task_name]
        optimizer_class = self.optimizers[optimizer_name]
        
        runner = task_class(
            optimizer_class=optimizer_class,
            optimizer_kwargs={'lr': lr},
            batch_size=batch_size
        )
        
        history = runner.run(epochs=epochs)
        metrics = calculate_ml_metrics(history['train'], history['val'])
        
        # Optional: Generate and save plot
        # fig = plot_ml_curves([history['train']['accuracy']], [history['val']['accuracy']], [optimizer_name])
        # fig.savefig(f"{task_name}_{optimizer_name}_curves.png")
        
        return metrics

    def list_benchmarks(self):
        """List available benchmarks."""
        return list(self.benchmarks.keys())

    def list_optimizers(self):
        """List available optimizers."""
        return list(self.optimizers.keys())

    def list_ml_tasks(self):
        """List available ML tasks."""
        return list(self.ml_tasks.keys())

if __name__ == "__main__":
    engine = Engine()
    
    print("Available Benchmarks:", engine.list_benchmarks())
    print("Available Optimizers:", engine.list_optimizers())
    print("Available ML Tasks:", engine.list_ml_tasks())
    
    # Example usage for benchmarks
    benchmark_name = 'Himmelblau'
    optimizer_name = 'Adam'
    initial_guess = np.random.uniform(-5, 5, size=2)
    benchmark_metrics = engine.run_benchmark(benchmark_name, optimizer_name, initial_guess)
    print(f"\nBenchmark Metrics for {benchmark_name} with {optimizer_name}: {benchmark_metrics}")
    
    # Example usage for ML tasks
    ml_task_name = 'MNIST'
    ml_metrics = engine.run_ml_task(ml_task_name, optimizer_name, epochs=2)
    print(f"\nML Task Metrics for {ml_task_name} with {optimizer_name}: {ml_metrics}")
