add_executable(cpszg_2d_v1_3 cpszg_2d_v1.3.cu)
add_executable(decpszg_2d_v1_3 decpszg_2d_v1.3.cu)

target_include_directories(
  cpszg_2d_v1_3
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(
  decpszg_2d_v1_3
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  cpszg_2d_v1_3
  PRIVATE
    factz::cuda_options
    factz_optimized_huffman
    factz::lorenzo
    FTK::libftk
)

target_link_libraries(
  decpszg_2d_v1_3
  PRIVATE
    factz::cuda_options
    factz_optimized_huffman
    factz::lorenzo
)

set_target_properties(
  cpszg_2d_v1_3 decpszg_2d_v1_3
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS cpszg_2d_v1_3 decpszg_2d_v1_3 RUNTIME DESTINATION bin)
