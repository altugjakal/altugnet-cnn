from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 class labels
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def get_data_loaders(batch_size=32):
    """
    Load and return CIFAR-10 training and testing data loaders.
    
    Args:
        batch_size: Batch size for training loader (default: 32)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(
        root="data", 
        download=True, 
        train=True, 
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="data", 
        download=True, 
        train=False, 
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    return train_loader, test_loader