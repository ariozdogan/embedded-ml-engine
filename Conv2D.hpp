#pragma once

#include "Tensor.hpp"

class Conv2D {
  public:
    Conv2D(int in_channels, int out_channels);

  private:
    Tensor kernel; // shape: (out_channels, in_channels, 3, 3)
    Tensor bias; // shape: (out_channels)
};