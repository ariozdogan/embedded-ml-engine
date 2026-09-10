#include "Tensor.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

Tensor::Tensor(std::vector<int> shape) {
  this->shape = shape;
  
  size_t shape_length = shape.size();

  this->strides.resize(shape_length);
  this->strides[shape_length-1] = 1;

  for (int i=shape_length-2; i>=0; i--) {
    this->strides[i] = this->strides[i+1] * shape[i+1];
  }

  int total_size = 1;
  for (int i=0; i<shape_length; i++) {
    total_size *= shape[i];
  }

  this->data.resize(total_size);
}

const float& Tensor::at(std::vector<int> indices) const {
  int flat_index = 0;

  for (int i = 0; i < indices.size(); i++) {
    flat_index += indices[i] * strides[i];
  }
  return data[flat_index];
}

float& Tensor::at(std::vector<int> indices) {
  return const_cast<float&>(static_cast<const Tensor&>(*this).at(indices));
}

Tensor Tensor::matmul(const Tensor& tensor_b) const {
  assert(this->shape[0] == tensor_b.shape[1]);

  int rows_a = this->shape[0];
  int cols_a = this->shape[1];
  int cols_b = tensor_b.shape[1];

  Tensor result({rows_a, cols_b});

  for (int i=0; i<rows_a; ++i) {
    for (int j=0; j<cols_b; ++j) {
      for (int k=0; k<cols_a; ++k) {
        result.at({i, j}) += this->at({i, k}) * tensor_b.at({k, j});
      }
    }
  }

  return result;
}

Tensor Tensor::add(const Tensor& tensor_b) const {
  assert(this->shape == tensor_b.shape);

  Tensor result(this->shape);

  for (size_t i=0; i<this->data.size(); ++i) {
    result.data[i] = this->data[i] + tensor_b.data[i];
  }

  return result;
}

void Tensor::print() const {
  int rows = this->shape[0];
  int cols = this->shape[1];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << this->at({i, j}) << "\t";
    }
    std::cout << std::endl;
  }
}
  
void Tensor::fill(float value) {
  for (size_t i=0; i<this->data.size(); ++i) {
    this->data[i] = value;
  }
}

std::vector<int> Tensor::getShape() const {
  return shape;
}

Tensor Tensor::relu() const {
  Tensor result(this->shape);

  for (size_t i=0; i<this->data.size(); ++i) {
    result.data[i] = std::max(0.0f, this->data[i]);
  }

  return result;
}

Tensor Tensor::softmax() const {
  Tensor result(this->shape);

  for (size_t row=0; row<this->shape[0]; ++row) {
    float max_value = 0;
    float exp_sum = 0;

    for (size_t col=0; col<this->shape[1]; ++col) {
      if (this->data[row*shape[1] + col] > max_value) {
        max_value = this->data[row*shape[1] + col];
      }
    }
    for (size_t col=0; col<this->shape[1]; ++col) {
      result.data[row*shape[1] + col] = std::exp(this->data[row*shape[1] + col] - max_value);
      exp_sum += result.data[row*shape[1] + col];
    }
    for (size_t col=0; col<this->shape[1]; ++col) {
      result.data[row*shape[1] + col] = result.data[row*shape[1] + col] / exp_sum;
    }
  }

  return result;
}