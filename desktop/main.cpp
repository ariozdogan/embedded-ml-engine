#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include "Conv2D.hpp"
#include "WeightLoader.hpp"
#include "Model.hpp"

int main() {
  Model model;
  std::ifstream image_file("../../data/test_image.bin", std::ios::binary);
  Tensor input = readTensor(image_file);
  Tensor output = model.forward(input);
  output.print();


  return 0;
}