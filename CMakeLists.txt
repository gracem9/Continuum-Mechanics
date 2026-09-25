cmake_minimum_required(VERSION 3.18...3.31)
project(elasticity_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++17, not -std=gnu++17

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Python + pybind11. When built through pip, scikit-build-core provides both;
# for a standalone build pass -Dpybind11_DIR=$(python -m pybind11 --cmakedir).
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Floating-point policy. -ffp-contract=off stops the compiler from fusing a*b+c
# into a single FMA instruction: FMA rounds once instead of twice, so results
# would drift from NumPy, which rounds every operation. We also deliberately do
# NOT use -ffast-math, which would let the compiler reorder arithmetic.
set(EK_EXTRA_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(EK_EXTRA_FLAGS -ffp-contract=off)
endif()

# --- Pure C++ kernel (no Python dependency) ---------------------------------
add_library(elasticity_core STATIC cpp/elasticity.cpp)
target_include_directories(elasticity_core PUBLIC cpp)
# SYSTEM: treat the vendored header as third-party, so our warning flags don't
# report warnings from code we don't own.
target_include_directories(elasticity_core SYSTEM PRIVATE third_party/pocketfft)
target_compile_definitions(elasticity_core PRIVATE
  POCKETFFT_NO_MULTITHREADING
  POCKETFFT_CACHE_SIZE=16)
target_compile_options(elasticity_core PRIVATE ${EK_EXTRA_FLAGS}
  $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra>)
# A static library linked into a shared Python module must be position-independent.
set_target_properties(elasticity_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Build metadata, so the benchmark can report how the module was built ----
string(TOUPPER "${CMAKE_BUILD_TYPE}" EK_BUILD_TYPE_UPPER)
string(JOIN " " EK_CXX_FLAGS
  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${EK_BUILD_TYPE_UPPER}} -std=c++17 ${EK_EXTRA_FLAGS})
string(STRIP "${EK_CXX_FLAGS}" EK_CXX_FLAGS)
set(EK_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(EK_FFT "pocketfft (vendored, commit c90e55b), single-threaded, real-to-complex")
configure_file(cpp/build_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.hpp @ONLY)

# --- Python extension module --------------------------------------------------
pybind11_add_module(_elasticity_cpp cpp/bindings.cpp)
target_link_libraries(_elasticity_cpp PRIVATE elasticity_core)
target_include_directories(_elasticity_cpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Installs the compiled module at the top level of site-packages, so
# `import _elasticity_cpp` works from anywhere, including src/elasticity_utils.py.
install(TARGETS _elasticity_cpp LIBRARY DESTINATION .)
