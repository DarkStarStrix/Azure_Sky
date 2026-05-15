"""
Plotting utilities for visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_surface(benchmark, paths, optimizer_names):
    """
    Plot a 3D surface of a benchmark function with optimization paths.
    
    Args:
        benchmark (BaseBenchmark): The benchmark instance.
        paths (list): List of paths (arrays of points) for each optimizer.
        optimizer_names (list): List of optimizer names.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate surface
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = benchmark.f(np.array([X[i, j], Y[i, j]]))
            
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    
    # Plot paths
    for path, name in zip(paths, optimizer_names):
        path = np.array(path)
        if path.shape[1] >= 2:
            z_path = [benchmark.f(p) for p in path]
            ax.plot(path[:, 0], path[:, 1], z_path, label=name, marker='o', markersize=2)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.legend()
    plt.close(fig)
    return fig

def plot_ml_curves(train_data, val_data, optimizer_names, metric='Accuracy'):
    """
    Plot learning curves for ML tasks.
    
    Args:
        train_data (list): List of training histories for each optimizer.
        val_data (list): List of validation histories for each optimizer.
        optimizer_names (list): List of optimizer names.
        metric (str): The metric being plotted (e.g., 'Accuracy', 'Loss').
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig = plt.figure(figsize=(10, 6))
    
    for t, v, name in zip(train_data, val_data, optimizer_names):
        epochs = range(1, len(t) + 1)
        plt.plot(epochs, t, label=f'{name} Train')
        plt.plot(epochs, v, '--', label=f'{name} Val')
        
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.close(fig)
    return fig
