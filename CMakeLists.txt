cmake_minimum_required(VERSION 3.21)

project(sz VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

option(FACTZ_BUILD_CUDA "Build CUDA implementations" ON)
option(FACTZ_BUILD_CUDA_NAIVE "Build the naive CUDA cpSZ-offline implementation" ON)
option(FACTZ_BUILD_CUDA_V1 "Build the V1 CUDA cpSZ-offline implementation" ON)
option(FACTZ_BUILD_CUDA_V1_1 "Build the V1.1 CUDA cpSZ-offline implementation" OFF)
option(FACTZ_BUILD_CUDA_V1_2 "Build the V1.2 CUDA cpSZ-offline implementation" OFF)
option(FACTZ_BUILD_CUDA_V1_3 "Build the V1.3 CUDA cpSZ-offline implementation" OFF)
option(FACTZ_BUILD_CUDA_V1_4 "Build the V1.4 CUDA cpSZ-offline implementation" OFF)
option(FACTZ_BUILD_CUDA_V1_5 "Build the V1.5 GPU-resident CUDA cpSZ-offline implementation" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(zstd REQUIRED)
find_package(FTK REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ftk/install/lib/cmake/FTK")

if(FACTZ_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(
      FATAL_ERROR
      "No CUDA compiler was found. Set CMAKE_CUDA_COMPILER or configure "
      "with -DFACTZ_BUILD_CUDA=OFF."
    )
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(
      CMAKE_CUDA_ARCHITECTURES "70;75;80;86"
      CACHE STRING "CUDA architectures used by FaCTz implementations" FORCE
    )
  endif()

  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/cusz/CMakeLists.txt")
    message(FATAL_ERROR "Missing vendored cuSZ source under external/cusz")
  endif()
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/cusz_compat/include/kernel/lrz/lproto.hh")
    message(FATAL_ERROR "Missing cuSZ compatibility headers under external/cusz_compat")
  endif()

  # CUDA versions share the vendored cuSZ/PHF core while keeping their own
  # pipeline sources in version-specific subdirectories.
  set(PSZ_BUILD_EXAMPLES OFF)
  set(PSZ_BUILD_MAIN_LIBRARY OFF)
  set(PSZ_BUILD_PYBINDING OFF)
  set(PSZ_CMAKE_ACTIVATE_LC OFF)
  set(CMAKE_DISABLE_FIND_PACKAGE_PORTABLE TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_FZG TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_UTILS TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_PHF TRUE)
  set(_FACTZ_BUILD_TESTING "${BUILD_TESTING}")
  set(BUILD_TESTING OFF)
  add_subdirectory(external/cusz "${PROJECT_BINARY_DIR}/external/cusz" EXCLUDE_FROM_ALL)
  set(BUILD_TESTING "${_FACTZ_BUILD_TESTING}")
  unset(_FACTZ_BUILD_TESTING)
endif()

add_subdirectory(src)
if(FACTZ_BUILD_CUDA)
  add_subdirectory(src/cuda)
endif()
add_subdirectory(test)
