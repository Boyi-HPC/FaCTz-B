/**
 * @file l21.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D5965FDA_3E90_4AC4_A53B_8439817D7F1C
#define D5965FDA_3E90_4AC4_A53B_8439817D7F1C

#include <stdint.h>

#include "cusz/type.h"
#include "mem/compact.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    void* out_outlier, double const eb, uint16_t const radius,
    float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    double const eb, uint16_t const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* eb_list, uint16_t const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq, typename EqEb>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__tile_eb(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    const EqEb* tile_eb, T threshold, uint16_t const radius,
    float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    double const eb, int const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_x_lorenzo_nd__eb_list(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq, typename EqEb>
pszerror GPU_PROTO_x_lorenzo_nd__tile_eb(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    const EqEb* tile_eb, int tile_dim, T threshold, int const radius,
    float* time_elapsed, void* stream);

// Chunk-based 1D Lorenzo: K chunks per row, all parallel
template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_chunk1d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* eb_list, uint16_t const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_x_lorenzo_chunk1d__eb_list(
    Eq* in_eq, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream);

// 1D row-wise Lorenzo: all rows parallel, pred = left neighbor
template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_row1d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* eb_list, uint16_t const radius,
    float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_x_lorenzo_row1d__eb_list(
    Eq* in_eq, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream);

// Correct variable-EB Lorenzo: predict in float space, sequential scan
template <typename T, typename Eq>
pszerror GPU_PROTO_correct_c_lorenzo_2d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* decomp_buf, T* eb_list, uint16_t const radius,
    float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_correct_x_lorenzo_2d__eb_list(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream);

}  // namespace psz::cuhip

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp::proto {
template <typename T, typename E>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb,
    int const radius, E* const eq, void* _outlier, float* time_elapsed,
    void* stream);

template <typename T, typename E>
pszerror GPU_x_lorenzo_nd(
    E* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb,
    int const radius, T* xdata, float* time_elapsed, void* stream);

}  // namespace psz::dpcpp::proto

#endif

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
