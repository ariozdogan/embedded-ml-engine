#pragma once

#include "Tensor.hpp"

class DenseLayer {
  public:
    DenseLayer(int in_features, int out_features);
    Tensor forward(const Tensor& input) const;
    void loadWeights(Tensor newWeight, Tensor newBias);

  private:
    Tensor W;
    Tensor bias;
};