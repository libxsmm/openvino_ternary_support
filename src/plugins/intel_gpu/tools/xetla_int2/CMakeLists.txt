# Standalone build: point OpenVINO_DIR at the OpenVINO build directory.
#   cmake -B build -DOpenVINO_DIR=<ov-build> -DCMAKE_CXX_COMPILER=icpx && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(xetla_int2_tools CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenVINO REQUIRED)
add_executable(bench_llm bench_llm.cpp)
target_link_libraries(bench_llm PRIVATE openvino::runtime)
