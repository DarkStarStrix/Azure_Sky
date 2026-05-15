"""
Usage examples for the AzureSky optimizer.
"""
import torch
import torch.nn as nn
from Backend.optimizers.azure_optim import Azure

class SimpleModel(nn.Module):
    """A simple two-layer model for demonstration purposes."""
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(10, 5)
        self.classifier = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.base(x))
        return self.classifier(x)

def example_basic_model():
    """Example 1: Basic usage with model.parameters()."""
    model = SimpleModel()
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()
    
    optimizer = Azure(model.parameters())
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"[Example 1] Loss: {loss.item():.4f}")

def example_parameter_groups():
    """Example 2: Parameter groups with different learning rates."""
    model = SimpleModel()
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()
    
    optimizer = Azure([
        {'params': model.base.parameters(), 'lr': 1e-2},
        {'params': model.classifier.parameters()}
    ])
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"[Example 2] Loss: {loss.item():.4f}")

def example_benchmark_optimization():
    """Example 3: Optimizing a mathematical benchmark function."""
    from Backend.Benchmarks.Himmelblau import Himmelblau
    
    benchmark = Himmelblau()
    x = torch.tensor([3.0, 2.0], requires_grad=True)
    
    optimizer = Azure([x], lr=0.01, sa_steps=200)
    
    for step in range(500):
        optimizer.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"[Example 3] Step {step}: Loss = {loss.item():.6f}, x = {x.data.numpy()}")

if __name__ == "__main__":
    print("=== AzureSky Optimizer Usage Examples ===\n")
    example_basic_model()
    example_parameter_groups()
    example_benchmark_optimization()
