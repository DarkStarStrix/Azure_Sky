"""
CIFAR-10 ML Task Runner.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Runner:
    """Runner for CIFAR-10 training and evaluation."""
    
    def __init__(self, optimizer_class, optimizer_kwargs=None, batch_size=64):
        """
        Initialize the runner.
        
        Args:
            optimizer_class (type): The class of the optimizer to use.
            optimizer_kwargs (dict): Keyword arguments for the optimizer.
            batch_size (int): Batch size for DataLoader.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        optimizer_kwargs = optimizer_kwargs or {'lr': 0.001}
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        self.train_loader, self.test_loader = self._load_data(batch_size)
        
    def _load_data(self, batch_size):
        """Load CIFAR-10 dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader

    def run(self, epochs=10):
        """
        Run the training loop.
        
        Args:
            epochs (int): Number of epochs to train.
            
        Returns:
            dict: Training history containing train and validation metrics.
        """
        history = {
            'train': {'loss': [], 'accuracy': []},
            'val': {'loss': [], 'accuracy': []}
        }
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            history['train']['loss'].append(train_loss / len(self.train_loader))
            history['train']['accuracy'].append(100. * correct / total)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
            history['val']['loss'].append(val_loss / len(self.test_loader))
            history['val']['accuracy'].append(100. * correct / total)
            
        return history
