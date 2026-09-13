#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include "Conv2D.hpp"

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

  // forward test
  Tensor input1({1, 2});  
  input1.at({0, 0}) = 1.0f;
  input1.at({0, 1}) = 2.0f;
  DenseLayer layer(2, 2);
  Tensor output = layer.forward(input1);
  output.print();

  // softmax test
  Tensor input2({1, 3});
  input2.at({0, 0}) = 1.0f;
  input2.at({0, 1}) = 2.0f;
  input2.at({0, 2}) = 3.0f;
  Tensor softmax = input2.softmax();
  softmax.print();

  // conv2d test
  Tensor input3({1, 1, 4, 4});
  Conv2D layer1(1,1);

  input3.at({0, 0, 0, 0}) = 1.0f;
  input3.at({0, 0, 0, 1}) = 2.0f;
  input3.at({0, 0, 0, 2}) = 3.0f;
  input3.at({0, 0, 0, 3}) = 4.0f;

  input3.at({0, 0, 1, 0}) = 5.0f;
  input3.at({0, 0, 1, 1}) = 6.0f;
  input3.at({0, 0, 1, 2}) = 7.0f;
  input3.at({0, 0, 1, 3}) = 8.0f;

  input3.at({0, 0, 2, 0}) = 9.0f;
  input3.at({0, 0, 2, 1}) = 10.0f;
  input3.at({0, 0, 2, 2}) = 11.0f;
  input3.at({0, 0, 2, 3}) = 12.0f;

  input3.at({0, 0, 3, 0}) = 13.0f;
  input3.at({0, 0, 3, 1}) = 14.0f;
  input3.at({0, 0, 3, 2}) = 15.0f;
  input3.at({0, 0, 3, 3}) = 16.0f;

  Tensor conv2d = layer1.forward(input3);
  conv2d.print();

  // maxpool test
  Tensor input4({1, 1, 4, 4});

  input4.at({0, 0, 0, 0}) = 1.0f;
  input4.at({0, 0, 0, 1}) = 2.0f;
  input4.at({0, 0, 0, 2}) = 3.0f;
  input4.at({0, 0, 0, 3}) = 4.0f;

  input4.at({0, 0, 1, 0}) = 5.0f;
  input4.at({0, 0, 1, 1}) = 6.0f;
  input4.at({0, 0, 1, 2}) = 7.0f;
  input4.at({0, 0, 1, 3}) = 8.0f;

  input4.at({0, 0, 2, 0}) = 9.0f;
  input4.at({0, 0, 2, 1}) = 10.0f;
  input4.at({0, 0, 2, 2}) = 11.0f;
  input4.at({0, 0, 2, 3}) = 12.0f;

  input4.at({0, 0, 3, 0}) = 13.0f;
  input4.at({0, 0, 3, 1}) = 14.0f;
  input4.at({0, 0, 3, 2}) = 15.0f;
  input4.at({0, 0, 3, 3}) = 16.0f;

  Tensor maxpool_output = input4.maxpool();
  maxpool_output.print();


  return 0;
}