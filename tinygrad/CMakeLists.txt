cmake_minimum_required(VERSION 3.16)

project(VampireGrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(VampireGrad
    src/main.cpp
    src/tensor.cpp
    src/matmul.cpp
    src/model_export.cpp
    src/tensor_autograd.cpp
)

target_include_directories(VampireGrad PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)