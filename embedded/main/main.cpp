#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include "Conv2D.hpp"
#include "WeightLoader.hpp"
#include "Model.hpp"

#include "esp_timer.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps.h"

extern const uint8_t weights_bin_start[] asm("_binary_weights_bin_start"); // points to first byte of the embedded data
extern const uint8_t weights_bin_end[] asm("_binary_weights_bin_end"); // points to last byte

extern const uint8_t binary_test_1_start[] asm("_binary_test_1_bin_start");
extern const uint8_t binary_test_1_end[] asm("_binary_test_1_bin_end");

void test_pipeline() {
  Model model;
  
  size_t offset = 0;
  Tensor input = readTensorFromMemory(binary_test_1_start, offset);

  int label = *reinterpret_cast<const int*>(binary_test_1_start + offset);

  Tensor output = model.forward(input);

  int predicted_class = 0;
  float max_value = output.at({0, 0});

  for (int j = 1; j < 10; ++j) {
    float value = output.at({0, j});

    if (value > max_value) {
      max_value = value;
      predicted_class = j;
    }
  }

  printf("predicted: %d, true label: %d\n", predicted_class, label);
}

void test_latency() {
  Model model;
  size_t offset = 0;

  Tensor input = readTensorFromMemory(binary_test_1_start, offset);

  int64_t start = esp_timer_get_time();
  Tensor output = model.forward(input);
  int64_t end = esp_timer_get_time();
  int64_t elapsed = end - start; // microseconds

  printf("elapsed time: %lld microseconds \n", elapsed);
}

extern "C" void app_main(void) {
  //test_pipeline(); // load single image to test pipeline is working

  //test_latency(); // test latency for one image

size_t free_before = esp_get_free_heap_size();

Model model;
size_t offset = 0;
Tensor input = readTensorFromMemory(binary_test_1_start, offset);

size_t min_free = esp_get_minimum_free_heap_size();

size_t free_after_construct = esp_get_free_heap_size();

Tensor output = model.forward(input);
size_t free_after_forward = esp_get_free_heap_size();

printf("construction used: %d bytes\n", free_before - free_after_construct);
printf("forward pass used: %d bytes\n", free_after_construct - free_after_forward);
printf("minimum free heap ever seen: %ld bytes\n", esp_get_minimum_free_heap_size());
printf("total heap size: %d bytes\n", heap_caps_get_total_size(MALLOC_CAP_DEFAULT));
printf("peak memory used: %d bytes\n", free_before - min_free);

}