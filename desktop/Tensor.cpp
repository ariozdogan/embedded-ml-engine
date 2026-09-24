#include "Tensor.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

 // constructs tensor with zeros of inputted shape
Tensor::Tensor(std::vector<int> shape) {
  this->shape = shape;
  
  size_t shape_length = shape.size();

  this->strides.resize(shape_length);

  // data is a flat 1D array in memory
  // strides[i] records how many positions must be skipped when moving a step along dimension i
  this->strides[shape_length-1] = 1;

  // each stride is the dimension after it multiplied by the current dimension
  for (int i=shape_length-2; i>=0; i--) {
    this->strides[i] = this->strides[i+1] * shape[i+1];
  }

  int total_size = 1;
  for (int i=0; i<shape_length; i++) {
    total_size *= shape[i]; // total size is the product of all the dimension sizes
  }

  this->data.resize(total_size);
}

// converts inputed multi-dimensional coordinates into a single flat position using strides
// for reading only
const float& Tensor::at(std::vector<int> indices) const {
  int flat_index = 0;

  for (int i = 0; i < indices.size(); i++) {
    flat_index += indices[i] * strides[i];
  }
  return data[flat_index];
}

// for reading and writing
float& Tensor::at(std::vector<int> indices) {
  return const_cast<float&>(static_cast<const Tensor&>(*this).at(indices));
}

Tensor Tensor::matmul(const Tensor& tensor_b) const {
  assert(this->shape[1] == tensor_b.shape[0]); // the inner dimensions of the two matrices must be equal, ex. (2x3)(3x4) 

  int rows_a = this->shape[0];
  int cols_a = this->shape[1];
  int cols_b = tensor_b.shape[1];

  Tensor result({rows_a, cols_b});

  for (int i=0; i<rows_a; ++i) {
    for (int j=0; j<cols_b; ++j) {
      for (int k=0; k<cols_a; ++k) {
        result.at({i, j}) += this->at({i, k}) * tensor_b.at({k, j}); // += because constructor initializes a 0 matrix 
      }
    }
  }

  return result;
}

Tensor Tensor::add(const Tensor& tensor_b) const {
  assert(this->shape == tensor_b.shape); // both matrix shapes must be equal

  Tensor result(this->shape);

  for (size_t i=0; i<this->data.size(); ++i) {
    result.data[i] = this->data[i] + tensor_b.data[i];
  }

  return result;
}

// used for debugging
// prints shape and all values in the matrix
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

// fills tensor with same value
void Tensor::fill(float value) {
  for (size_t i=0; i<this->data.size(); ++i) {
    this->data[i] = value;
  }
}

// returns copy of shape (read only)
std::vector<int> Tensor::getShape() const {
  return shape;
}

// Rectified Linear Unit (ReLU)
// outputs original number if positive, outputs zero if original number is negative
Tensor Tensor::relu() const {
  Tensor result(this->shape);

  for (size_t i=0; i<this->data.size(); ++i) {
    result.data[i] = std::max(0.0f, this->data[i]);
  }

  return result;
}

// softmax normalizes each row of the tensor into a probability distribution between [0, 1] all summing up to 1
Tensor Tensor::softmax() const {
  Tensor result(this->shape);

  // loops through each row
  for (size_t row=0; row<this->shape[0]; ++row) {
    float max_value = 0;
    float exp_sum = 0;

    // first loop determines the max
    for (size_t col=0; col<this->shape[1]; ++col) {
      if (this->data[row*shape[1] + col] > max_value) {
        max_value = this->data[row*shape[1] + col];
      }
    }
    // compute exp(x - max) and accumulate their sum
    for (size_t col=0; col<this->shape[1]; ++col) {
      result.data[row*shape[1] + col] = std::exp(this->data[row*shape[1] + col] - max_value);
      exp_sum += result.data[row*shape[1] + col];
    }
    // divide each value by the accumulated sum
    for (size_t col=0; col<this->shape[1]; ++col) {
      result.data[row*shape[1] + col] = result.data[row*shape[1] + col] / exp_sum;
    }
  }

  return result;
}

// condenses (batch, channels, height, width) to (batch, in_channels) for dense layer to compute with a flat row per example
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

// maxpool with 2x2 matrix and non-overlapping window
// height/2 and width/2, batch and channel count unchanged
Tensor Tensor::maxpool() const {
  int batch = this->shape[0];
  int in_channels = this->shape[1];
  int height = this->shape[2];
  int width = this->shape[3];

  int out_channels = in_channels;
  int out_height = height / 2;
  int out_width = width / 2;

  Tensor result({batch, out_channels, out_height, out_width});

  // b, out_c, out_row, out_col decide which value in output matrix is being computed
  for (int b = 0; b < batch; ++b) {
    for (int out_c = 0; out_c < out_channels; ++out_c) {
      for (int out_row = 0; out_row < out_height; ++out_row) {
        for (int out_col = 0; out_col < out_width; ++out_col) {
          
          // position of top left corner of 2x2 maxpooling window
          int first_row = out_row * 2 + 0;
          int first_col = out_col * 2 + 0;

          // initialize max_value for each new window
          float max_value = this->at({b, out_c, first_row, first_col});

          // scan 2x2 window storing the max value
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

          result.at({b, out_c, out_row, out_col}) = max_value; // add max value to the new matrix

        }
      }
    }
  }

  return result;
}

// writes value at flat position in data
// exists because data is private
void Tensor::setFlat(size_t index, float value) {
  data[index] = value;
}

// returns total value count
size_t Tensor::size() const {
  return data.size();
}

// returns most probable output in output layer (0-9)
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