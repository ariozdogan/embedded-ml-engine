#include <vector>

class Tensor {
  public:
    Tensor(std::vector<int> shape);
    float& at(std::vector<int> indices);
    const float& at(std::vector<int> indices) const;
    Tensor matmul(const Tensor& tensor_b) const;
    void print() const;
  
  private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
};