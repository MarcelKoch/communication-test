cmake_minimum_required(VERSION 3.21.0)

project(commtest LANGUAGES CXX)


include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    message(STATUS "Enabling HIP support")
    enable_language(HIP)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT CMAKE_HIP_COMPILER)
    message(STATUS "Enabling CUDA support")
    enable_language(CUDA)
endif ()

if(NOT CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No device language enabled. Please make sure that either HIP or CUDA are available")
endif ()

find_package(Ginkgo REQUIRED)
find_package(benchmark REQUIRED)
find_package(MPI REQUIRED)

add_library(device_libs INTERFACE)

if(CMAKE_HIP_COMPILER)
    find_package(hip REQUIRED)
    find_package(rccl REQUIRED)

    target_link_libraries(device_libs INTERFACE roc::rccl hip::host)
    target_compile_definitions(device_libs INTERFACE __NCCL_INC=<rccl/rccl.h> __EXEC=HipExecutor)
endif ()
if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED NCCL_HOME AND DEFINED ENV{NCCL_HOME})
        set(NCCL_HOME $ENV{NCCL_HOME})
    endif ()
    if(NOT DEFINED NCCL_HOME)
        message(FATAL_ERROR "Please specifiy the NCCL home directory by either setting NCCL_HOME as an environment "
                "variable or through cmake.")
    endif ()

    include(FindPkgConfig)
    pkg_check_modules(nccl REQUIRED IMPORTED_TARGET nccl)

    target_link_libraries(device_libs INTERFACE PkgConfig::nccl)
    target_compile_definitions(device_libs INTERFACE __NCCL_INC=<nccl.h> __EXEC=CudaExecutor)
endif ()

add_executable(main main.cpp)
target_link_libraries(main Ginkgo::ginkgo device_libs MPI::MPI_CXX)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks Ginkgo::ginkgo benchmark::benchmark device_libs MPI::MPI_CXX)
