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

image, label = test_data[0]
print("true label", label)

image = image.unsqueeze(0)

with open("../data/test_image.bin", "wb") as f:
  write_tensor(f, image)