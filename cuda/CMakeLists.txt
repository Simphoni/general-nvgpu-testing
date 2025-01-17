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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -UNDEBUG -O2")


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Wall -UNDEBUG --ptxas-options=-v --use_fast_math")

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

file(GLOB_RECURSE BEAM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_library(beam_C SHARED ${BEAM_SOURCES})
set_target_properties(beam_C PROPERTIES PREFIX "" SUFFIX ${Python3_EXTENSION_SUFFIX})
target_include_directories(beam_C PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(beam_C PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include)
target_include_directories(beam_C PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/ThunderKittens/include)
target_compile_definitions(beam_C PUBLIC INTERFACE_NAME=beam_C)
target_link_libraries(beam_C ${TORCH_LIBRARIES} pybind11::module)

set_target_properties(beam_C PROPERTIES INSTALL_RPATH "\$ORIGIN;${CMAKE_CURRENT_BINARY_DIR}")

set(arce_install_dir "${CMAKE_SOURCE_DIR}/python/beam/_C")

install(TARGETS beam_C LIBRARY DESTINATION ${arce_install_dir} COMPONENT torch_ext)

set(generate_stub TRUE)
# NOTE: calling with `python3 -m pybind11_stubgen` will cause different behavior
# which we wouldn't like
set(stub_generator "pybind11-stubgen")
set(stub_output_dir "${CMAKE_SOURCE_DIR}/python/stubs")

if (generate_stub)
  add_custom_target(
    stubgen
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${stub_output_dir}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}/python ${stub_generator} beam -o ${stub_output_dir}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${stub_output_dir}/beam ${CMAKE_SOURCE_DIR}/python/beam
    COMMENT "Generating Python stubs with ${stub_generator}"
  )
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build . --target stubgen)")
endif()