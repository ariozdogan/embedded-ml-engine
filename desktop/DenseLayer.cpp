#include "DenseLayer.hpp"
#include <iostream>


// fully-connected (dense) layer 
DenseLayer::DenseLayer(int in_features, int out_features)
    : W({in_features, out_features}), bias({out_features}) { 

    // placeholder constants before filled with actual weights in Model
    W.fill(0.1);
    bias.fill(0.1);
}

Tensor DenseLayer::forward(const Tensor& input) const {
  Tensor result = input.matmul(W); // shape: (batch, out_features)

  std::vector<int> shape = result.getShape();
  int batch = shape[0];
  int out_features = shape[1];

  for (int i=0; i<batch; ++i) {
    for (int j=0; j<out_features; ++j) {
      result.at({i, j}) = result.at({i, j}) + bias.at({j});
    }
  }

  return result; // returns tensor with (feature * weight) + bias in each index
}

void DenseLayer::loadWeights(Tensor newWeight, Tensor newBias) {
  W = newWeight;
  bias = newBias;
}