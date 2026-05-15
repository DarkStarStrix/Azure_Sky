"""
Smoke tests for the AzureSky codebase.
"""
import torch
import numpy as np
from Backend.Benchmarks.Himmelblau import Himmelblau
from Backend.Benchmarks.Adjiman import Adjiman
from Backend.Benchmarks.Ackley import AckleyN2
from Backend.Benchmarks.Brent import Brent
from Backend.optimizers.azure_optim import Azure
from Backend.optimizers.adam import AdamOptimizer
from Backend.optimizers.SGD import SGDOptimizer
from Backend.optimizers.RMSprop import RMSpropOptimizer
from Metrics import calculate_benchmark_metrics, calculate_ml_metrics

def test_himmelblau_azure():
    benchmark = Himmelblau()
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = Azure([x], lr=0.05, sa_steps=100)
    for _ in range(200):
        opt.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        opt.step()
        benchmark.add_to_path(x)
        benchmark.add_loss_value(loss)
    benchmark.set_global_min([3.0, 2.0])
    metrics = benchmark.get_metrics()
    print(f"[Azure/Himmelblau] loss={metrics['final_loss']:.6f}, dist={metrics['distance']:.4f}")
    assert metrics['final_loss'] < 10.0, "Loss should decrease"

def test_himmelblau_adam():
    benchmark = Himmelblau()
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = AdamOptimizer([x], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        opt.step()
        benchmark.add_to_path(x)
        benchmark.add_loss_value(loss)
    benchmark.set_global_min([3.0, 2.0])
    metrics = benchmark.get_metrics()
    print(f"[Adam/Himmelblau]  loss={metrics['final_loss']:.6f}, dist={metrics['distance']:.4f}")
    assert metrics['final_loss'] < 10.0, "Loss should decrease"

def test_brent_sgd():
    benchmark = Brent()
    x = torch.tensor([0.5], requires_grad=True)
    opt = SGDOptimizer([x], lr=0.001, momentum=0.9)
    for _ in range(300):
        opt.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        opt.step()
        benchmark.add_to_path(x)
        benchmark.add_loss_value(loss)
    benchmark.set_global_min([1.0])
    metrics = benchmark.get_metrics()
    print(f"[SGD/Brent]        loss={metrics['final_loss']:.6f}, dist={metrics['distance']:.4f}")

def test_ackley_rmsprop():
    benchmark = AckleyN2()
    x = torch.zeros(10, requires_grad=True)
    opt = RMSpropOptimizer([x], lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        opt.step()
        benchmark.add_to_path(x)
        benchmark.add_loss_value(loss)
    benchmark.set_global_min([0.0] * 10)
    metrics = benchmark.get_metrics()
    print(f"[RMSprop/Ackley]   loss={metrics['final_loss']:.6f}, dist={metrics['distance']:.4f}")

def test_ml_metrics():
    train_history = {'loss': [1.0, 0.8, 0.6], 'accuracy': [60.0, 70.0, 80.0]}
    val_history = {'loss': [1.1, 0.9, 0.7], 'accuracy': [58.0, 67.0, 75.0]}
    metrics = calculate_ml_metrics(train_history, val_history)
    assert metrics['best_epoch'] == 3
    assert metrics['final_train_acc'] == 80.0
    print(f"[ML Metrics]       best_epoch={metrics['best_epoch']}, final_val_acc={metrics['final_val_acc']}")

if __name__ == "__main__":
    print("=== AzureSky Smoke Tests ===\n")
    test_himmelblau_azure()
    test_himmelblau_adam()
    test_brent_sgd()
    test_ackley_rmsprop()
    test_ml_metrics()
    print("\nAll tests passed.")
