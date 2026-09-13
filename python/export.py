import torch
import struct
from model import TinyMLModel

model = TinyMLModel()
model.load_state_dict(torch.load("../data/model_weights.pth"))

def write_tensor(f, tensor):
  data = tensor.detach().numpy()
  shape = data.shape

  # struct converts a py int into its raw binary representation, matching a C++ int

  f.write(struct.pack('i', len(shape))) # number of dimensions

  for dim in shape:
    f.write(struct.pack('i', dim)) # each dimension's size

  f.write(data.tobytes())

  with open("../data/model_weights.pth")as f:
    write_tensor(f, model.conv1.weight)
    write_tensor(f, model.conv1.bias)
    write_tensor(f, model.fc1.weight)
    write_tensor(f, model.fc1.bias)