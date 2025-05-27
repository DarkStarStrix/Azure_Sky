import torch
import torch.nn as nn
from azure_optimizer import Azure  # Assuming the above class is in a module named azure_optimizer

# Define a simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(10, 5)
        self.classifier = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.base(x))
        return self.classifier(x)

# Initialize model and sample variables
model = SimpleModel()
var1 = torch.nn.Parameter(torch.randn(2, 2))
var2 = torch.nn.Parameter(torch.randn(2, 2))
inputs = torch.randn(32, 10)
targets = torch.randint(0, 2, (32,))
criterion = nn.CrossEntropyLoss()

# Example 1: Basic usage with model.parameters()
optimizer = Azure(model.parameters())
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# Example 2: List of parameters
optimizer = Azure([var1, var2])
optimizer.zero_grad()
loss = criterion(var1 @ var2, torch.zeros_like(var1 @ var2))
loss.backward()
optimizer.step()

# Example 3: Named parameters
optimizer = Azure(model.named_parameters())
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# Example 4: Named parameters in a list (invalid, will be handled by the class)
optimizer = Azure([('layer0', var1), ('layer1', var2)])  # The class converts this to a parameter list
optimizer.zero_grad()
loss = criterion(var1 @ var2, torch.zeros_like(var1 @ var2))
loss.backward()
optimizer.step()

# Example 5: Parameter groups with different learning rates
optimizer = Azure([
    {'params': model.base.parameters(), 'lr': 1e-2},
    {'params': model.classifier.parameters()}
])
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# Example 6: Parameter groups with named parameters
optimizer = Azure([
    {'params': model.base.named_parameters(), 'lr': 1e-2},
    {'params': model.classifier.named_parameters()}
])
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
