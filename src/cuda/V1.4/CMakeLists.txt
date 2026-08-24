add_executable(cpszg_2d_v1_4 cpszg_2d_v1.4.cu)

target_include_directories(
  cpszg_2d_v1_4
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(
  cpszg_2d_v1_4
  PRIVATE
    factz::cuda_options
    factz_optimized_huffman
    factz_ans
    factz::lorenzo
    FTK::libftk
)

set_target_properties(
  cpszg_2d_v1_4
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS cpszg_2d_v1_4 RUNTIME DESTINATION bin)
