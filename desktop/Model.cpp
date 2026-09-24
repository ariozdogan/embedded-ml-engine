#include "Model.hpp"
#include "WeightLoader.hpp"

Model::Model() : conv1(1, 8), dense1(1352, 10) { // (1 input channel, 8 output), (1352 flattened features, 10 output classes)
  std::ifstream file("../../data/weights.bin", std::ios::binary); 

  // importing trained weights from PyTorch
  Tensor conv_kernel = readTensor(file);
  Tensor conv_bias = readTensor(file);
  Tensor dense_W = readTensor(file);
  Tensor dense_bias = readTensor(file);

  // filler weights are replaced with the trained weights
  conv1.loadWeights(conv_kernel, conv_bias);
  dense1.loadWeights(dense_W, dense_bias);
  }

// assembles ReLU -> maxpool -> flatten -> dense -> softmax pipeline
Tensor Model::forward(const Tensor& input) const {
  Tensor x = conv1.forward(input);

  x = x.relu();
  x = x.maxpool();
  x = x.flatten();
  x = dense1.forward(x);
  x = x.softmax();

  return x;
}