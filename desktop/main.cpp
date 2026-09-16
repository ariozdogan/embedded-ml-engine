#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include "Conv2D.hpp"
#include "WeightLoader.hpp"
#include "Model.hpp"

int main() {
  Model model;
  int correct_count = 0;

  for (int i=1; i<50; ++i) {
    std::ifstream image_file("../../data/test_" + std::to_string(i) + ".bin", std::ios::binary);
    Tensor input = readTensor(image_file);

    int label;
    image_file.read(reinterpret_cast<char*>(&label), sizeof(int));

    Tensor output = model.forward(input);
    int output_index = output.argmax();

    if (output_index == label) {
      correct_count += 1;
    }
  }

  std::cout << "accuracy: " << (float)correct_count / 50 * 100 << "%" << std::endl;

  return 0;
}