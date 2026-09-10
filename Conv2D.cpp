#include "Conv2D.hpp"

Conv2D::Conv2D(int in_channels, int out_channels) 
  : kernel({out_channels, in_channels, 3, 3}), bias({out_channels}) {
    
    kernel.fill(0.1);
    bias.fill(0.1);
} 