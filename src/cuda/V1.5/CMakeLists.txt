add_executable(cpszg_2d_v1_5 cpszg_2d.cu)
add_executable(decpszg_2d_v1_5 decpszg_2d.cu)

target_include_directories(
  cpszg_2d_v1_5
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(
  decpszg_2d_v1_5
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  cpszg_2d_v1_5
  PRIVATE
    factz::cuda_options
    factz_optimized_huffman
    factz_ans
    factz::lorenzo
    FTK::libftk
)

target_link_libraries(
  decpszg_2d_v1_5
  PRIVATE
    factz::cuda_options
    factz_optimized_huffman
    factz_ans
    factz::lorenzo
)

set_target_properties(
  cpszg_2d_v1_5 decpszg_2d_v1_5
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS cpszg_2d_v1_5 decpszg_2d_v1_5 RUNTIME DESTINATION bin)
