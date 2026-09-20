#include "Model.hpp"
#include "WeightLoader.hpp"

extern const uint8_t weights_bin_start[] asm("_binary_weights_bin_start"); // points to first byte of the embedded data
extern const uint8_t weights_bin_end[] asm("_binary_weights_bin_end"); // points to last byte

Model::Model() : conv1(1, 8), dense1(1352, 10) { // (1 input channel, 8 output), (1568 flattened features, 10 output classes)
  size_t offset = 0;

  Tensor conv_kernel = readTensorFromMemory(weights_bin_start, offset);
  Tensor conv_bias = readTensorFromMemory(weights_bin_start, offset);
  Tensor dense_W = readTensorFromMemory(weights_bin_start, offset);
  Tensor dense_bias = readTensorFromMemory(weights_bin_start, offset);

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