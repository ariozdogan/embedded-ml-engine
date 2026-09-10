#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "DenseLayer.hpp"

int main() {
  // testing methods

  Tensor a({2, 2});
  a.at({0, 0}) = 1.0f; a.at({0, 1}) = 2.0f;
  a.at({1, 0}) = 3.0f; a.at({1, 1}) = 4.0f;

  Tensor b({2, 2});
  b.at({0, 0}) = 5.0f; b.at({0, 1}) = 6.0f;
  b.at({1, 0}) = 7.0f; b.at({1, 1}) = 8.0f;

  Tensor result = a.matmul(b);
  result.print();


  // Tensor input({1, 2});
  // input.at({0, 0}) = 1.0f;
  // input.at({0, 1}) = 2.0f;
  // DenseLayer layer(2, 2);
  // Tensor output = layer.forward(input);
  // output.print();


  Tensor input({1, 3});
  input.at({0, 0}) = 1.0f;
  input.at({0, 1}) = 2.0f;
  input.at({0, 2}) = 3.0f;
  Tensor softmax = input.softmax();
  softmax.print();

  return 0;
}