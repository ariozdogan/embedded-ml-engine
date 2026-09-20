idf_component_register(SRCS "main.cpp" "Tensor.cpp" "DenseLayer.cpp" "Conv2D.cpp" "WeightLoader.cpp" "Model.cpp"
                        INCLUDE_DIRS "."
                        EMBED_FILES "weights.bin" "test_1.bin")