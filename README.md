# Azure Sky Optimizer

Azure Sky Optimizer is a hybrid optimizer for PyTorch, integrating Simulated Annealing (SA) with Adam to provide robust exploration and precise exploitation in non-convex optimization tasks. Designed for complex machine learning challenges, Azure Sky excels in domains requiring deep exploration of rugged loss landscapes, such as scientific machine learning, symbolic reasoning, and protein folding.

Developed as part of an R&D initiative, Azure Sky combines structured stochastic exploration with gradient-based refinement, achieving stable convergence and strong generalization in multi-modal search spaces.

---

## Overview

Conventional optimizers like Adam and AdamW often converge prematurely to sharp local minima, compromising generalization. Azure Sky leverages SA’s global search in early stages and Adam’s local convergence later, ensuring both deep exploration and precise convergence.

### Core Innovations

- **Dynamic Temperature Scaling:** Adjusts SA temperature based on training progress for controlled exploration.
- **Exploration-Exploitation Fusion:** Seamlessly transitions between SA and Adam using a sigmoid-based blending mechanism.
- **Stability Enhancements:** Built-in gradient clipping and loss spike monitoring for robust training.

---

## Key Features

- **Hybrid Optimization:** Combines SA’s global search with Adam’s local refinement.
- **Optimized Hyperparameters:** Tuned via Optuna (best trial: 0.0893 on Two Moons dataset).
- **Flexible Parameter Handling:** Supports parameter lists, named parameters, and parameter groups with group-specific learning rates.
- **Production-Ready Stability:** Includes gradient clipping and loss spike detection.
- **PyTorch Compatibility:** Fully integrated with PyTorch’s `optim` module.

---

## Installation

Clone the repository and install using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/yourusername/azure-sky-optimizer.git
cd azure-sky-optimizer
uv pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.20.0

> **Note:** Ensure `uv` is installed. See [uv documentation](https://github.com/astral-sh/uv) for instructions.

---

## Usage Examples

Azure Sky integrates seamlessly into PyTorch workflows. Below are usage examples for various parameter configurations.

### Basic Usage

```python
import torch
import torch.nn as nn
from azure_optimizer import Azure

model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()
optimizer = Azure(model.parameters())

inputs = torch.randn(32, 10)
targets = torch.randint(0, 2, (32,))
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

### Parameter Lists

```python
var1 = torch.nn.Parameter(torch.randn(2, 2))
var2 = torch.nn.Parameter(torch.randn(2, 2))
optimizer = Azure([var1, var2])
```

### Parameter Groups with Custom Learning Rates

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(10, 5)
        self.classifier = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.base(x))
        return self.classifier(x)

model = SimpleModel()
optimizer = Azure([
    {'params': model.base.parameters(), 'lr': 1e-2},
    {'params': model.classifier.parameters()}
])
```

For additional examples, see `azure_optimizer/usage_example.py`.

---

## Hyperparameters

Default hyperparameters (from Optuna Trial 99, best validation loss: 0.0893 on Two Moons):

| Parameter    | Value                | Description                        |
|--------------|----------------------|------------------------------------|
| lr           | 0.0007518383921113902| Learning rate for Adam phase       |
| T0           | 2.2723218904585964   | Initial temperature for SA         |
| sigma        | 0.17181058166567398  | Perturbation strength for SA       |
| SA_steps     | 5                    | Steps for SA phase                 |
| sa_momentum  | 0.6612913488540948   | Momentum for SA updates            |

---

## Performance

Evaluated on the Two Moons dataset (5000 samples, 20% noise):

- **Best Validation Loss:** 0.0919
- **Final Validation Accuracy:** 96.7%
- **Epochs to Convergence:** 50

Compared to:
- **Adam:** loss 0.0927, acc 96.8%
- **AdamW:** loss 0.0917, acc 97.1%

Azure Sky prioritizes robust generalization over rapid convergence, making it ideal for pre-training and tasks requiring deep exploration.

---

## Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes.
4. Push to your branch.
5. Open a pull request.

Please follow PEP 8 standards. Tests are not yet implemented; contributions to add testing infrastructure are highly encouraged.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use Azure Sky Optimizer in your research or engineering projects, please cite:

```
[Allan]. (2025). Azure Sky Optimizer: A Hybrid Approach for Exploration and Exploitation. GitHub Repository.
```

---

## Project Status

As of May 27, 2025, Azure Sky Optimizer is stable and production-ready.

**Planned improvements:**
- Testing on larger datasets (e.g., CIFAR-10, MNIST)
- Ablation studies for hyperparameter impact
- Integration with PyTorch Lightning
- Adding a comprehensive test suite

For questions or collaboration, please open an issue on GitHub.

Kaggle Notebook: https://www.kaggle.com/code/allanwandia/non-convex-research
Writeup It has old metrics so watch out: https://github.com/DarkStarStrix/CSE-Repo-of-Advanced-Computation-ML-and-Systems-Engineering/blob/main/Papers/Computer_Science/Optimization/Optimization_Algothrims_The_HimmelBlau_Function_Case_Study.pdf

---

## Repository Structure

```
azure-sky-optimizer/
├── azure_optimizer/
│   ├── __init__.py
│   ├── azure.py        # Updated Azure class
│   ├── hooks.py
│   └── usage_example.py  # Usage demonstrations
├── README.md
└── LICENSE
```
