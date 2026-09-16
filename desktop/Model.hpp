#pragma once

#include "Tensor.hpp"
#include "Conv2D.hpp"
#include "DenseLayer.hpp"
#include <fstream>

class Model {
  public:
    Model();
    Tensor forward(const Tensor& input) const;

  private:
    Conv2D conv1;
    DenseLayer dense1;
};