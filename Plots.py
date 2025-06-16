import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_surface(benchmark, paths, optimizer_names):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[benchmark.f(np.array([xi, yi])) for xi in x] for yi in y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    for path, name in zip(paths, optimizer_names):
        ax.plot(path[:, 0], path[:, 1], [benchmark.f(p) for p in path], label=name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.legend()
    plt.close()
    return fig

def plot_ml_curves(train_data, val_data, optimizer_names, metric='Accuracy'):
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
    plt.close()
    return fig