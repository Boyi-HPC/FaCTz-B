add_library(factz_cuda_options INTERFACE)
add_library(factz::cuda_options ALIAS factz_cuda_options)

target_compile_features(
  factz_cuda_options
  INTERFACE
    cxx_std_17
    cuda_std_17
)

target_compile_definitions(factz_cuda_options INTERFACE PSZ_USE_CUDA=1)

target_compile_options(
  factz_cuda_options
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-Wno-deprecated-declarations>
)

target_include_directories(
  factz_cuda_options
  INTERFACE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(factz_cuda_options INTERFACE CUDA::cudart)

add_library(
  factz_lorenzo
  STATIC
  cusz/lproto_c.cu
  cusz/lproto_x.cu
)
add_library(factz::lorenzo ALIAS factz_lorenzo)

target_include_directories(
  factz_lorenzo
  PUBLIC
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cusz"
)

target_link_libraries(factz_lorenzo PUBLIC factz::cuda_options)

set_target_properties(
  factz_lorenzo
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

if(FACTZ_BUILD_CUDA_V1 OR FACTZ_BUILD_CUDA_V1_1)
  add_library(
    factz_v1_huffman
    STATIC
    naive/gpu_huffman_bridge_no_opt_speed.cu
    V1/gpu_huffman_u2_bridge.cu
  )
  target_link_libraries(
    factz_v1_huffman
    PUBLIC
      factz::cuda_options
      PHF::phf_cu
      psz_cu_core
  )
  set_target_properties(
    factz_v1_huffman
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
endif()

if(FACTZ_BUILD_CUDA_V1_2 OR FACTZ_BUILD_CUDA_V1_3 OR FACTZ_BUILD_CUDA_V1_4 OR FACTZ_BUILD_CUDA_V1_5)
  add_library(factz_optimized_huffman STATIC gpu_huffman_bridge.cu)
  target_include_directories(
    factz_optimized_huffman
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(
    factz_optimized_huffman
    PUBLIC
      factz::cuda_options
      PHF::phf_cu
      psz_cu_core
  )
  set_target_properties(
    factz_optimized_huffman
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
endif()

if(FACTZ_BUILD_CUDA_V1_4 OR FACTZ_BUILD_CUDA_V1_5)
  find_package(nvcomp REQUIRED)
  get_filename_component(_FACTZ_NVCOMP_LIBRARY_DIR "${nvcomp_LIBRARY}" DIRECTORY)
  list(APPEND CMAKE_BUILD_RPATH "${_FACTZ_NVCOMP_LIBRARY_DIR}")

  add_library(factz_ans STATIC gpu_ans_bridge.cu)
  target_include_directories(factz_ans PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(
    factz_ans
    PUBLIC
      factz::cuda_options
      nvcomp::nvcomp
  )
  set_target_properties(
    factz_ans
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )

  add_executable(factz_decpszg_2d tools/decpszg_2d.cu)
  target_include_directories(
    factz_decpszg_2d
    PRIVATE
      "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
      "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(
    factz_decpszg_2d
    PRIVATE
      factz::cuda_options
      factz_optimized_huffman
      factz_ans
      factz::lorenzo
  )

  add_executable(factz_verify_cpszg_2d tools/verify_cpszg_2d.cpp)
  target_include_directories(
    factz_verify_cpszg_2d
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
  )
  target_link_libraries(factz_verify_cpszg_2d PRIVATE FTK::libftk)
endif()

if(FACTZ_BUILD_CUDA_NAIVE)
  add_subdirectory(naive)
endif()

if(FACTZ_BUILD_CUDA_V1)
  add_subdirectory(V1)
endif()

if(FACTZ_BUILD_CUDA_V1_1)
  add_subdirectory(V1.1)
endif()

if(FACTZ_BUILD_CUDA_V1_2)
  add_subdirectory(V1.2)
endif()

if(FACTZ_BUILD_CUDA_V1_3)
  add_subdirectory(V1.3)
endif()

if(FACTZ_BUILD_CUDA_V1_4)
  add_subdirectory(V1.4)
endif()

if(FACTZ_BUILD_CUDA_V1_5)
  add_subdirectory(V1.5)
endif()
