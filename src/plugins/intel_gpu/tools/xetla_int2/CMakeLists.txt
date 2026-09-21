# Standalone build: point OpenVINO_DIR at the OpenVINO build directory.
#   cmake -B build -DOpenVINO_DIR=<ov-build> -DCMAKE_CXX_COMPILER=icpx && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(xetla_int2_tools CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenVINO REQUIRED)

# bench_llm         8B, stateful (indirect SDPA)
# paged_bench_llm   8B, SDPAToPagedAttention
# *_27b             the VLM decoder: separate embedding model, rank-3 mrope
# paged_serve_llm_27b  batched (continuous-batching) generation from a request file, for eval harnesses
foreach(tool bench_llm paged_bench_llm bench_llm_27b paged_bench_llm_27b paged_serve_llm_27b)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE openvino::runtime)
endforeach()
