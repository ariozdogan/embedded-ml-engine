from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_data = datasets.MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="../data", train=False  , download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)