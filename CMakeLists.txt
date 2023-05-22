cmake_minimum_required(VERSION 3.21.0)

project(commtest LANGUAGES CXX)

set(enabled_device_lang OFF)

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    message(STATUS "Enabling HIP support")
    enable_language(HIP)
    set(enabled_device_lang HIP)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT enabled_device_lang)
    message(STATUS "Enabling CUDA support")
    enable_language(CUDA)
    set(enabled_device_lang CUDA)
endif ()

if(NOT enabled_device_lang)
    message(FATAL_ERROR "No device language enabled. Please make sure that either HIP or CUDA are available")
endif ()

find_package(Ginkgo REQUIRED)
find_package(benchmark REQUIRED)
find_package(MPI REQUIRED)

add_library(device_libs INTERFACE)

if(enabled_device_lang STREQUAL HIP)
    find_package(hip REQUIRED)
    find_package(rccl REQUIRED)

    target_link_libraries(device_libs INTERFACE roc::rccl hip::host)
    target_compile_definitions(device_libs INTERFACE __NCCL_INC=<rccl/rccl.h> __EXEC=HipExecutor)
endif ()
if(enabled_decive_lang STREQUAL CUDA)
    target_link_libraries(device_libs INTERFACE nccl)
    target_compile_definitions(device_libs INTERFACE __NCCL_INC=<nccl.h> __EXEC=CudaExecutor)
endif ()

add_executable(main main.cpp)
target_link_libraries(main Ginkgo::ginkgo device_libs MPI::MPI_CXX)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks Ginkgo::ginkgo benchmark::benchmark device_libs MPI::MPI_CXX)
