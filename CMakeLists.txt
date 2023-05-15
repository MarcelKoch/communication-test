cmake_minimum_required(VERSION 3.21.0)

project(commtest LANGUAGES CXX HIP)

find_package(Ginkgo REQUIRED)
find_package(hip REQUIRED)
find_package(rccl REQUIRED)
find_package(benchmark REQUIRED)
find_package(MPI REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main Ginkgo::ginkgo rccl hip::host)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks Ginkgo::ginkgo rccl hip::host benchmark::benchmark)
