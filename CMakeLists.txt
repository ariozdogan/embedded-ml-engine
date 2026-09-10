cmake_minimum_required(VERSION 3.10)
project(TinyMLInferenceEngine)

add_executable(main main.cpp Tensor.cpp DenseLayer.cpp)