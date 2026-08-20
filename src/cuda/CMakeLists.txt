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

if(FACTZ_BUILD_CUDA_NAIVE)
  add_subdirectory(naive)
endif()
