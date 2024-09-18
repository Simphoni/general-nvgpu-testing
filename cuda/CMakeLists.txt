cmake_minimum_required(VERSION 3.22)
set(DEFAULT_BUILD_TYPE "RelWithDebInfo")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python3 COMPONENTS Interpreter Development)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE Python3_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Trying to locate packages from ${Python3_SITE_PACKAGES}")

# set(CAFFE2_USE_CUDNN ON)
# set(CAFFE2_USE_CUSPARSELT ON)

if (EXISTS ${Python3_SITE_PACKAGES})
    if (EXISTS ${Python3_SITE_PACKAGES}/torch)
        set(Torch_ROOT "${Python3_SITE_PACKAGES}/torch")
        # set(CUDNN_LIBRARY_PATH "${Torch_ROOT}/lib/libcudnn.so.8")
    endif()
    if (EXISTS ${Python3_SITE_PACKAGES}/nvidia)
        # set(CUDNN_INCLUDE_DIR "${Python3_SITE_PACKAGES}/nvidia/cudnn/include")
    endif()
    if (EXISTS ${Python3_SITE_PACKAGES}/pybind11)
        set(pybind11_ROOT "${Python3_SITE_PACKAGES}/pybind11")
    endif()
endif()

set(TORCH_CUDA_ARCH_LIST "8.0")
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
# libtorch_python is for pybind interface
append_torchlib_if_found(torch_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -UNDEBUG -O2")


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Wall -UNDEBUG --ptxas-options=-v")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(acre SHARED interface.cc cublas.cc cutlass.cu)
set_target_properties(acre PROPERTIES PREFIX "" SUFFIX ${Python3_EXTENSION_SUFFIX})
target_include_directories(acre PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(acre PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include)
target_compile_definitions(acre PRIVATE INTERFACE_NAME=acre)
target_link_libraries(acre ${TORCH_LIBRARIES} pybind11::module)

set_target_properties(acre PROPERTIES INSTALL_RPATH "\$ORIGIN;${CMAKE_CURRENT_BINARY_DIR}")

set(arce_install_dir "${CMAKE_SOURCE_DIR}/python")

install(TARGETS acre LIBRARY DESTINATION ${arce_install_dir} COMPONENT torch_extension)