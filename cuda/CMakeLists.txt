cmake_minimum_required(VERSION 3.22)
project(CublasBaseline LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main cublas.cu)
target_link_libraries(main PRIVATE ${CUDA_LIBRARIES})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")