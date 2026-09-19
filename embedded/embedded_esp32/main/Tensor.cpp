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
  assert(this->shape[1] == tensor_b.shape[0]);

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
  std::cout << "Shape: (";
  for (size_t i=0; i<this->shape.size(); ++i) {
    std::cout << this->shape[i];
    if (i < this->shape.size() - 1) {
      std::cout <<", ";
    }
  }
  std::cout << ")   " << std::endl;

  for (size_t i=0; i<this->data.size(); ++i) {
    std::cout << this->data[i] << "  ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
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

Tensor Tensor::flatten() const {
  int batch = this->shape[0];
  int channels = this->shape[1];
  int height = this->shape[2];
  int width = this->shape[3];

  std::vector<int> newShape = {batch, channels * height * width};
  Tensor result(newShape);

  result.data = this->data;

  return result;
}

Tensor Tensor::maxpool() const {
  int batch = this->shape[0];
  int in_channels = this->shape[1];
  int height = this->shape[2];
  int width = this->shape[3];

  int out_channels = in_channels;
  int out_height = height / 2; // 2x2 matrix
  int out_width = width / 2;

  Tensor result({batch, out_channels, out_height, out_width});

  for (int b = 0; b < batch; ++b) {
    for (int out_c = 0; out_c < out_channels; ++out_c) {
      for (int out_row = 0; out_row < out_height; ++out_row) {
        for (int out_col = 0; out_col < out_width; ++out_col) {
          
          int first_row = out_row * 2 + 0;
          int first_col = out_col * 2 + 0;
          float max_value = this->at({b, out_c, first_row, first_col});

          for (int k_r = 0; k_r < 2; ++k_r) {
            for (int k_c = 0; k_c < 2; ++k_c) {

              int input_row = out_row * 2 + k_r;
              int input_col = out_col * 2 + k_c;

              float value = this->at({b, out_c, input_row, input_col});

              if (value > max_value) {
                  max_value = value;
              }
            }
          }

          result.at({b, out_c, out_row, out_col}) = max_value;

        }
      }
    }
  }

  return result;
}

void Tensor::setFlat(size_t index, float value) {
  data[index] = value;
}

size_t Tensor::size() const {
  return data.size();
}

int Tensor::argmax() const {
  float max = this->at({0, 0});
  int return_index = 0;

  for (int i = 1; i < 10; ++i) {
    float value = this->at({0, i});
    if (value > max) {
      max = value;
      return_index = i;
    }
  }

  return return_index;
}