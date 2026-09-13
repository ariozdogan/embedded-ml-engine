#include "Conv2D.hpp"

Conv2D::Conv2D(int in_channels, int out_channels) 
  : kernel({out_channels, in_channels, 3, 3}), bias({out_channels}) {
    
    kernel.fill(0.1);
    bias.fill(0.1);
} 

Tensor Conv2D::forward(const Tensor& input) const {
  int batch = input.getShape()[0];
  int in_channels = input.getShape()[1];
  int height = input.getShape()[2];
  int width = input.getShape()[3];

  int out_channels = kernel.getShape()[0];
  int out_height = height - 3 + 1;
  int out_width = width - 3 + 1;

  Tensor result({batch, out_channels, out_height, out_width});

  // convolution process
  for (int b=0; b<batch; ++b) {                              // batch size 
    for (int out_c=0; out_c<out_channels; ++out_c) {         // output channels: each filter in layer
      for (int out_row=0; out_row<out_height; ++out_row) {   // output height: moves filter down image
        for (int out_col=0; out_col<out_width; ++out_col) {  // output width: moves filter across image

          float sum = 0.0f;

          for (int in_c=0; in_c<in_channels; ++in_c) {       // input channels: loops through color channels
            for (int k_r=0; k_r<3; ++k_r) {                  // kernel row: loops through vertical size
              for (int k_c=0; k_c<3; ++k_c) {                // kernel col: loops through horizontal size
                
                int input_row = out_row + k_r;
                int input_col = out_col + k_c;

                float input_value = input.at({b, in_c, input_row, input_col});
                float kernel_value = kernel.at({out_c, in_c, k_r, k_c});

                sum += (input_value * kernel_value);

              }
            }
          }

          sum += bias.at({out_c});
          result.at({b, out_c, out_row, out_col}) = sum;

        }
      }
    }
  }

  return result;
}