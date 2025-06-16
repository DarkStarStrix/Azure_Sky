# CIFAR Dataset implementation in Pytorch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_cifar10(batch_size=64, num_workers=2, download=True):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def run_cifar10():
    """Run CIFAR-10 dataset loading and basic iteration."""
    train_loader, test_loader = load_cifar10(batch_size=64, num_workers=2, download=True)

    # Example: Iterate through the training data
    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels}")
        break  # Remove this break to iterate through all batches

if __name__ == "__main__":
    train_loader, test_loader = load_cifar10(batch_size=64, num_workers=2, download=True)

    # Example: Iterate through the training data
    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels}")
        break  # Remove this break to iterate through all batches
