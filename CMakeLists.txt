cmake_minimum_required(VERSION 3.24)
project(dynamic_ot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # RTX 4080 (Ada Lovelace) = compute capability 8.9
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(ot_solver STATIC
    src/kernels_common.cu
    src/kernels_interp.cu
    src/kernels_divergence.cu
    src/kernels_prox_j.cu
    src/prox_c.cu
    src/prox_c_fft.cu
    src/prox_consensus.cu
    src/dr_iteration.cu
    src/solver.cu
)
target_include_directories(ot_solver PUBLIC include)
target_link_libraries(ot_solver PUBLIC CUDA::cudart CUDA::cufft)
target_compile_options(ot_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(ot_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
set_target_properties(ot_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(validation_gaussians test/validation_gaussians.cu)
target_link_libraries(validation_gaussians PRIVATE ot_solver)
set_target_properties(validation_gaussians PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optional visualization tooling
add_executable(export_density tools/export_density.cu)
target_link_libraries(export_density PRIVATE ot_solver)
set_target_properties(export_density PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

