add_executable(cpszg_2d_v1 cpszg_2d_v1.cu)

target_include_directories(
  cpszg_2d_v1
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  cpszg_2d_v1
  PRIVATE
    factz::cuda_options
    factz_v1_huffman
    factz::lorenzo
    FTK::libftk
)

set_target_properties(
  cpszg_2d_v1
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS cpszg_2d_v1 RUNTIME DESTINATION bin)
