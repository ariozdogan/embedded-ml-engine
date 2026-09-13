import data
import torch
from model import TinyMLModel

model = TinyMLModel()
model.load_state_dict(torch.load("../data/model_weights.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
correct_predictions = 0
total_items = 0

with torch.no_grad():
  for inputs, targets in data.test_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)

    predictions = torch.argmax(outputs, dim=1)

    correct_predictions += (predictions == targets).sum().item()
    total_items += targets.size(0)

accuracy = correct_predictions / total_items * 100
print(f"model accuracy: {accuracy:.2f}%")