#pragma once

#include "Tensor.hpp"
#include "WeightLoader.hpp"
#include <fstream>

Tensor readTensor(std::ifstream& file) {
  int num_dims;
  // reinterprets the memory address of the int* as a char* so read() accepts it
  file.read(reinterpret_cast<char*>(&num_dims), sizeof(int)); 

  std::vector<int> shape;
  for (int i=0; i<num_dims; ++i) {
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    shape.push_back(dim);
  }

  Tensor result(shape);

  for (size_t i=0; i<result.size(); ++i) {
    float value;
    file.read(reinterpret_cast<char*>(&value), sizeof(float));
    result.setFlat(i, value);
  }

  return result;
}