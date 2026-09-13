import data

image, labels = next(iter(data.train_loader))
print(image.shape)
print(labels.shape)