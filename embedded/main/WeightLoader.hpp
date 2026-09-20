#pragma once

#include "Tensor.hpp"
#include <cstdint>

Tensor readTensorFromMemory(const uint8_t* data, size_t& offset); // now the weights are stored locally. offset is total # of bytes consumed starting at the first byte in weights.bin
