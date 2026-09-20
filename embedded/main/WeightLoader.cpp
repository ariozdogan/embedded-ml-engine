#include "Tensor.hpp"
#include "WeightLoader.hpp"
#include <iostream>

Tensor readTensorFromMemory(const uint8_t* data, size_t& offset) {
  int num_dims = *reinterpret_cast<const int*>(data + offset); // starting address + number of bytes into data
  offset += sizeof(int); // advances position past the 4 bytes

  std::vector<int> shape;

  for (int i=0; i<num_dims; ++i) {
    int dim = *reinterpret_cast<const int*>(data + offset);
    offset += sizeof(int);
    shape.push_back(dim); // adds int to end of vector
  }

  Tensor result(shape);
  for (size_t i=0; i<result.size(); ++i) {
    float value = *reinterpret_cast<const float*>(data + offset); // weights and biases
    offset += sizeof(float);
    result.setFlat(i, value); // write value into result
  }

  return result;
}