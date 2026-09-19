# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_WITH_SYCL)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj")

# Sources usable when the plugin runs on the OpenCL runtime, i.e. SYCL kernels
# enqueued through OpenCL interop rather than a SYCL-native runtime.
set(SOURCES_OCL_RT
    "${CMAKE_CURRENT_SOURCE_DIR}/impl_example.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_xetla_int2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_gemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_dpas.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xetla/hadamard_fwht.cpp")

if(GPU_RT_TYPE STREQUAL "SYCL")
    set(EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
else()
    file(GLOB_RECURSE EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(REMOVE_ITEM EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
endif()

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    BYPASS
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
)

add_sycl_to_target(TARGET ${TARGET_NAME})

if(NOT DEFINED XETLA_ROOT)
    set(XETLA_ROOT "$ENV{XETLA_ROOT}")
endif()
if(XETLA_ROOT AND EXISTS "${XETLA_ROOT}/include/xetla.hpp")
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${XETLA_ROOT}/include")
    # XeTLA is C++20 and its ESIMD kernels need the VC backend; both are scoped to
    # the vendored kernel so the rest of the plugin is unaffected.
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_gemv.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_dpas.cpp"
        PROPERTIES COMPILE_OPTIONS
            "-std=c++20;-Wno-unusable-partial-specialization;-Xsycl-target-backend=-vc-codegen -doubleGRF")
else()
    message(FATAL_ERROR "XETLA_ROOT is not set or has no include/xetla.hpp; "
                        "needed by the XeTLA int2 FullyConnected impl")
endif()
