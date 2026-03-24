cmake_minimum_required(VERSION 3.18)
project(TensorProductVolumeOCCA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── Dependencies ──────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
find_package(OCCA REQUIRED)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  message(STATUS "CUDA toolkit found — enabling GPU paths (cuBLAS, OCCA/CUDA, native CUDA kernel)")
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  set(USE_CUDA TRUE)
else()
  message(STATUS "CUDA toolkit not found — building CPU-only")
  set(USE_CUDA FALSE)
endif()

# LAPACK / BLAS (needed for CPU dgemm path)
find_package(LAPACK REQUIRED)

# cuTENSOR — disabled for now
# if(USE_CUDA)
#   set(CUTENSOR_ROOT $ENV{CUTENSOR_ROOT})
#   include_directories(${CUTENSOR_ROOT}/include)
#   link_directories(${CUTENSOR_ROOT}/lib)
# endif()

# OKL kernel directory (compiled at runtime by OCCA)
get_filename_component(DGX3D_OKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/okl" ABSOLUTE)

# ── Executable ────────────────────────────────────────────────────────
if(USE_CUDA)
  add_executable(tensor_product_volume main.cpp Timer.cpp gemm.cpp TensorProductCUDA.cu)
  set_source_files_properties(TensorProductCUDA.cu PROPERTIES
    COMPILE_FLAGS "-O3 --use_fast_math"
  )
else()
  add_executable(tensor_product_volume main.cpp Timer.cpp gemm.cpp)
endif()

target_compile_definitions(tensor_product_volume PRIVATE
  DGX3D_OKL_DIR="${DGX3D_OKL_DIR}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tensor_product_volume PRIVATE -Wall -Wextra -O3)
endif()

# ── Linking ───────────────────────────────────────────────────────────
if(USE_CUDA)
  target_compile_definitions(tensor_product_volume PRIVATE DGX3D_HAS_CUDA)
  target_link_libraries(tensor_product_volume PUBLIC
    OCCA::libocca
    CUDA::cudart
    CUDA::cublas
    # cutensor          # disabled — uncomment when cuTENSOR is needed
    OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES}
  )
else()
  target_link_libraries(tensor_product_volume PUBLIC
    OCCA::libocca
    OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES}
  )
endif()
