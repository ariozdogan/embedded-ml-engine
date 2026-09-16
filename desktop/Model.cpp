#include "Model.hpp"
#include "WeightLoader.hpp"

Model::Model() : conv1(1, 8), dense1(1352, 10) { // (1 input channel, 8 output), (1568 flattened features, 10 output classes)
  std::ifstream file("../../data/weights.bin", std::ios::binary); 

  Tensor conv_kernel = readTensor(file);
  Tensor conv_bias = readTensor(file);
  Tensor dense_W = readTensor(file);
  Tensor dense_bias = readTensor(file);

  conv1.loadWeights(conv_kernel, conv_bias);
  dense1.loadWeights(dense_W, dense_bias);
  }

Tensor Model::forward(const Tensor& input) const {
  Tensor x = conv1.forward(input);

  x = x.relu();
  x = x.maxpool();
  x = x.flatten();
  x = dense1.forward(x);
  x = x.softmax();

  return x;
}