add_library(
  factz_naive_huffman
  STATIC
  gpu_huffman_bridge_no_opt_speed.cu
)

target_link_libraries(
  factz_naive_huffman
  PUBLIC
    factz::cuda_options
    PHF::phf_cu
    psz_cu_core
)

add_executable(cpszg_2d_no_opt_speed cpszg_2d_no_opt_speed.cu)

target_include_directories(
  cpszg_2d_no_opt_speed
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/cusz_compat/include"
)

target_link_libraries(
  cpszg_2d_no_opt_speed
  PRIVATE
    factz::cuda_options
    factz_naive_huffman
    factz::lorenzo
    FTK::libftk
)

set_target_properties(
  factz_naive_huffman
  cpszg_2d_no_opt_speed
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

install(TARGETS cpszg_2d_no_opt_speed RUNTIME DESTINATION bin)
