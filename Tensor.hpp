#pragma once

#include <vector>

class Tensor {
  public:
    Tensor(std::vector<int> shape);
    float& at(std::vector<int> indices);
    const float& at(std::vector<int> indices) const;
    Tensor matmul(const Tensor& tensor_b) const;
    Tensor add(const Tensor& tensor_b) const;
    void print() const;
    void fill(float value);
    std::vector<int> getShape() const; // returns copy of shape (read only)
    Tensor relu() const;
    Tensor softmax() const;
    Tensor flatten() const; 
    Tensor maxpool() const;

  private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
};