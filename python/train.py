import data
import torch
from model import TinyMLModel
import torch.nn as nn
import torch.optim as optim

model = TinyMLModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 20
model.train()

for epoch in range(epochs):
  running_loss = 0.0
  for batch_X, batch_y in data.train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad() # clear gradients

    outputs = model(batch_X) # forward pass

    loss = criterion(outputs, batch_y)

    loss.backward() # baackward pass

    optimizer.step() # update weights

    running_loss += loss.item()

  print(f"Epoch {epoch+1}: loss was {running_loss / len(data.train_loader)}")

torch.save(model.state_dict(), "../data/model_weigths.pth")