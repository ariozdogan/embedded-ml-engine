from torchvision import datasets, transforms
import struct

def write_tensor(f, tensor):
  data = tensor.detach().numpy()
  shape = data.shape

  f.write(struct.pack('i', len(shape)))
  for dim in shape:
    f.write(struct.pack('i', dim))
  f.write(data.tobytes())

test_data = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())

for i in range(50):
  image, label = test_data[i]
  image = image.unsqueeze(0)

  with open(f"../data/test_{i+1}.bin", "wb") as f:
    write_tensor(f, image)
    f.write(struct.pack('i', label))