import torch.nn as nn

class TinyMLModel(nn.module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1) 
    # conv2d output = input - kernel_size + 1 = 28 - 3 + 1 = 26
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    # maxpool out = input / 2 = 13
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=8*13*13, out_features=10) # digits are 0-9

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.fc1(x)

    return x