/**
 * @file lorenzo_proto.inl
 * @author Jiannan Tian
 * @brief (prototype) Dual-Eq Lorenzo method.
 * @version 0.2
 * @date 2019-09-23
 * (create) 2019-09-23 (rev) 2023-04-03
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include <cstddef>
#include <stdexcept>

#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/it_cuda.hh"
#include "utils/timer.hh"
#include "../../lorenzo_tile_dim.h"

namespace psz {

// Warp-aggregated atomic increment. All active lanes in a warp share ONE
// atomicAdd on the global counter, then each lane derives its unique slot from
// the ballot. Reduces atomic contention by up to 32x — critical when many
// threads emit outliers at the same instruction (e.g. chunk-boundary emission).
__device__ __forceinline__ uint32_t atomicAggInc(uint32_t* ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int lane = threadIdx.x & 31u;
  unsigned int rank = __popc(active & ((1u << lane) - 1u));
  uint32_t warp_old = 0;
  if (lane == (unsigned)leader) warp_old = atomicAdd(ctr, (uint32_t)change);
  warp_old = __shfl_sync(active, warp_old, leader);
  return warp_old + rank;
}

// easy algorithmic description

template <
    typename T, int TileDim = 256, typename Eq = uint16_t,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_1d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  // prequant (fp presence)
  if (id < data_len3.x) { data(0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0) - (t().x == 0 ? 0 : data(-1));
  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary1()) {  // postquant
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

template <
    typename T, int TileDim = 16, typename Eq = uint16_t,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_2d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;

  __shared__ T buf[TileDim][TileDim + 1];

  uint32_t y = threadIdx.y, x = threadIdx.x;
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  auto id = gid2();

  if (check_boundary2()) { data(0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +             // dist=1
                          (y > 0 ? data(0, -1) : 0) -             // dist=1
                          (x > 0 and y > 0 ? data(-1, -1) : 0));  // dist=2

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary2()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}


template <
    typename T, int TileDim = LORENZO_TILE_DIM, typename Eq = uint16_t,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_2d1l__eb_list(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, T* eb_list)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];
  // __shared__ T buf_decompressed[TileDim][TileDim + 1];

  uint32_t y = threadIdx.y, x = threadIdx.x;
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };
  // auto decompressed_data =[&](auto dx, auto dy) -> T& {
  //   return buf_decompressed[t().y + dy][t().x + dx];
  // };

  auto id = gid2();
  if (check_boundary2()) { 
    auto ebx2_r = 0.5 / eb_list[id];
    data(0, 0) = round(in_data[id] *  ebx2_r); //quantilize
    // decompressed_data(0, 0) = data(0, 0) * eb_list[id] * 2; //dequantilize
  }
  __syncthreads();
  T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +             // dist=1
                        (y > 0 ? data(0, -1) : 0) -             // dist=1
                        (x > 0 and y > 0 ? data(-1, -1) : 0));  // dist=2
  // T decompressed_delta = round((decompressed_data(0, 0) - ((x > 0 ? decompressed_data(-1, 0) : 0) +              // dist=1
  //                                      (y > 0 ? decompressed_data(0, -1) : 0) -                     // dist=1
  //                                      (x > 0 and y > 0 ? decompressed_data(-1, -1) : 0)))          // dist=2
  //                                                                                         /(2*eb_list[id]));  
  // if(id==4232079){
  //   printf("id:%d, data:%f, decompressed_data:%f, delta:%f, decompressed_delta:%f, eb_list:%f\n", id, data(0,0), decompressed_data(0,0), delta, decompressed_delta, eb_list[id]);
  // }
  bool quantizable = ((fabs(delta) < radius));
  T candidate = delta + radius;
  if (check_boundary2()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

template <
    typename T, int TileDim = LORENZO_TILE_DIM, typename Eq = uint16_t,
    typename EqEb = uint8_t, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_2d1l__tile_eb(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius,
    const EqEb* tile_eb, T threshold)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];
  __shared__ T ebx2_r;

  uint32_t y = threadIdx.y, x = threadIdx.x;
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  if (x == 0 && y == 0) {
    size_t tile_id = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int eb_id = (unsigned int)tile_eb[tile_id];
    T eb = (T)(1ULL << (2 * eb_id)) * threshold;
    ebx2_r = (T)0.5 / eb;
  }
  __syncthreads();

  auto id = gid2();
  if (check_boundary2()) data(0, 0) = round(in_data[id] * ebx2_r);
  __syncthreads();

  T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +
                          (y > 0 ? data(0, -1) : 0) -
                          (x > 0 && y > 0 ? data(-1, -1) : 0));
  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary2()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (!quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

template <
    typename T, int TileDim = 8, typename Eq = uint16_t,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_3d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim][TileDim + 1];

  auto z = t().z, y = t().y, x = t().x;
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  auto id = gid3();
  if (check_boundary3()) { data(0, 0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0, 0) -
            ((z > 0 and y > 0 and x > 0 ? data(-1, -1, -1) : 0)  // dist=3
             - (y > 0 and x > 0 ? data(-1, -1, 0) : 0)           // dist=2
             - (z > 0 and x > 0 ? data(-1, 0, -1) : 0)           //
             - (z > 0 and y > 0 ? data(0, -1, -1) : 0)           //
             + (x > 0 ? data(-1, 0, 0) : 0)                      // dist=1
             + (y > 0 ? data(0, -1, 0) : 0)                      //
             + (z > 0 ? data(0, 0, -1) : 0));                    //

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary3()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

// Chunk-based 1D Lorenzo compressor.
// Each thread handles one CHUNK of a row (row_id, chunk_id).
// Chunk boundary elements stored as exact outliers → O(K*r1) extra outliers
// (negligible for K=10). Within each chunk: sequential scan. Across chunks: parallel.
// This breaks the r2-length sequential chain into K×r1 independent K-length chains.
template <typename T, typename Eq = uint16_t,
          typename CompactVal = T, typename CompactIdx = uint32_t,
          typename CompactNum = uint32_t>
__global__ void KERNEL_c_lorenzo_chunk1d__eb_list(
    T* const in_data, dim3 const data_len3,
    Eq* const out_eq, CompactVal* const out_cval,
    CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, T* eb_list, uint32_t chunk_size)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t K = (r2 + chunk_size - 1) / chunk_size;  // chunks per row
  size_t chunk_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_id >= r1 * K) return;

  size_t row   = chunk_id / K;
  size_t chunk = chunk_id % K;
  size_t j0    = chunk * chunk_size;
  size_t j_end = (j0 + chunk_size < r2) ? j0 + chunk_size : r2;

  // Chunk boundary (non-first chunk, non-first column): store exact as outlier.
  // This gives decompressor a guaranteed-exact starting point for each chunk.
  double prev = 0.0;
  if (chunk > 0) {
    size_t bid = row * r2 + j0;
    out_eq[bid] = 0;
    uint32_t idx = atomicAggInc(out_cn);  // warp-aggregated: 1 atomic per warp
    out_cidx[idx] = (uint32_t)bid;
    out_cval[idx] = in_data[bid];
    prev = (double)in_data[bid];
    j0++;  // boundary handled; rest of chunk continues from j0+1
  }

  for (size_t j = j0; j < j_end; j++) {
    size_t id = row * r2 + j;
    double cur  = (double)in_data[id];
    double eb   = (double)eb_list[id];
    double pred = prev;

    // First column of entire row (chunk=0, j=0): no left neighbor → exact outlier
    if (chunk == 0 && j == 0) {
      out_eq[id] = 0;
      uint32_t idx = atomicAggInc(out_cn);
      out_cidx[idx] = (uint32_t)id;
      out_cval[idx] = in_data[id];
      prev = cur;
      continue;
    }

    if (eb > 0.0) {
      double diff  = cur - pred;
      double qdiff = fabs(diff) / eb + 1.0;
      if (qdiff < (double)(2 * radius)) {
        qdiff = (diff > 0.0) ? qdiff : -qdiff;
        int qi = (int)(qdiff / 2) + radius;
        double decomp = pred + (double)(qi - radius) * 2.0 * eb;
        if (fabs(decomp - cur) < eb) {
          out_eq[id] = (Eq)qi;
          prev = decomp;
          continue;
        }
      }
    }
    out_eq[id] = 0;
    uint32_t idx = atomicAggInc(out_cn);
    out_cidx[idx] = (uint32_t)id;
    out_cval[idx] = in_data[id];
    prev = cur;
  }
}

// 1D row-wise Lorenzo compressor: pred[i,j] = decomp[i,j-1].
// One thread per row; all rows run in parallel (no cross-row dependency).
// decomp_buf must be zero-initialized. Outliers store exact original float.
template <typename T, typename Eq = uint16_t,
          typename CompactVal = T, typename CompactIdx = uint32_t,
          typename CompactNum = uint32_t>
__global__ void KERNEL_c_lorenzo_row1d__eb_list(
    T* const in_data, dim3 const data_len3,
    Eq* const out_eq, CompactVal* const out_cval,
    CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, T* eb_list)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= r1) return;

  double prev = 0.0;  // d[row, -1] = 0

  for (size_t j = 0; j < r2; j++) {
    size_t id = row * r2 + j;
    double cur = (double)in_data[id];
    double eb  = (double)eb_list[id];
    double pred = prev;

    // First column: no left neighbor → store exact (avoids rare-code HF depth issue)
    if (j == 0) {
      out_eq[id] = 0;
      uint32_t idx = atomicAdd(out_cn, 1);
      out_cidx[idx] = (uint32_t)id;
      out_cval[idx] = in_data[id];
      prev = cur;
      continue;
    }

    if (eb > 0.0) {
      double diff  = cur - pred;
      double qdiff = fabs(diff) / eb + 1.0;
      if (qdiff < (double)(2 * radius)) {
        qdiff = (diff > 0.0) ? qdiff : -qdiff;
        int qi = (int)(qdiff / 2) + radius;
        double decomp = pred + (double)(qi - radius) * 2.0 * eb;
        if (fabs(decomp - cur) < eb) {
          out_eq[id] = (Eq)qi;
          prev = decomp;
          continue;
        }
      }
    }
    // Outlier: store exact original value
    out_eq[id] = 0;
    uint32_t idx = atomicAdd(out_cn, 1);
    out_cidx[idx] = (uint32_t)id;
    out_cval[idx] = in_data[id];
    prev = cur;
  }
}

// Correct variable-EB 2D Lorenzo compressor: predicts in float space.
// Sequential single-thread kernel; decomp_buf must be zero-initialized before call.
template <typename T, typename Eq = uint16_t,
          typename CompactVal = T, typename CompactIdx = uint32_t,
          typename CompactNum = uint32_t>
__global__ void KERNEL_correct_c_lorenzo_2d__eb_list(
    T* const in_data, dim3 const data_len3,
    Eq* const out_eq, CompactVal* const out_cval,
    CompactIdx* const out_cidx, CompactNum* const out_cn,
    T* decomp_buf, uint16_t const radius, T* eb_list)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  for (size_t i = 0; i < r1; i++) {
    for (size_t j = 0; j < r2; j++) {
      size_t id = i * r2 + j;
      T cur = in_data[id];
      T eb  = eb_list[id];

      double left    = (j > 0)           ? (double)decomp_buf[id - 1]      : 0.0;
      double top     = (i > 0)           ? (double)decomp_buf[id - r2]     : 0.0;
      double topleft = (i > 0 && j > 0) ? (double)decomp_buf[id - r2 - 1] : 0.0;
      double pred = left + top - topleft;

      if (eb > 0) {
        double diff  = (double)cur - pred;
        double qdiff = fabs(diff) / (double)eb + 1.0;
        if (qdiff < (double)(2 * radius)) {
          qdiff = (diff > 0) ? qdiff : -qdiff;
          int qi = (int)(qdiff / 2) + radius;
          T decomp = (T)(pred + (double)(qi - radius) * 2.0 * (double)eb);
          if (fabs((double)decomp - (double)cur) < (double)eb) {
            out_eq[id]      = (Eq)qi;
            decomp_buf[id]  = decomp;
            continue;
          }
        }
      }
      // Outlier: store original value directly (CPU-style, no pred arithmetic)
      out_eq[id] = 0;
      uint32_t idx = atomicAdd(out_cn, 1);
      out_cidx[idx] = (uint32_t)id;
      out_cval[idx] = cur;
      decomp_buf[id] = cur;
    }
  }
}

}  // namespace psz

namespace psz::cuhip {

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    void* out_outlier, double const eb, uint16_t const radius,
    float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (data_len3.z == 1 and data_len3.y == 1)
      return 1;
    else if (data_len3.z == 1 and data_len3.y != 1)
      return 2;
    else
      return 3;
  };

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;

  auto ot = (Compact*)out_outlier;

  constexpr auto Tile1D = dim3(256, 1, 1), Tile2D = dim3(16, 16, 1),
                 Tile3D = dim3(8, 8, 8);
  constexpr auto Block1D = dim3(256, 1, 1), Block2D = dim3(16, 16, 1),
                 Block3D = dim3(8, 8, 8);

  auto Grid1D = divide3(data_len3, Tile1D),
       Grid2D = divide3(data_len3, Tile2D),
       Grid3D = divide3(data_len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (ndim() == 1) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(),
            ot->num(), radius, ebx2_r);
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_2d1l<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(),
            ot->num(), radius, ebx2_r);
  }
  else if (ndim() == 3) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(),
            ot->num(), radius, ebx2_r);
  }
  else {
    throw std::runtime_error("Lorenzo only works for 123-D.");
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num, double const eb, uint16_t const radius,
    float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (data_len3.z == 1 and data_len3.y == 1)
      return 1;
    else if (data_len3.z == 1 and data_len3.y != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = dim3(256, 1, 1), Tile2D = dim3(16, 16, 1),
                 Tile3D = dim3(8, 8, 8);
  constexpr auto Block1D = dim3(256, 1, 1), Block2D = dim3(16, 16, 1),
                 Block3D = dim3(8, 8, 8);

  auto Grid1D = divide3(data_len3, Tile1D),
       Grid2D = divide3(data_len3, Tile2D),
       Grid3D = divide3(data_len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (ndim() == 1) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, ebx2_r);
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_2d1l<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, ebx2_r);
  }
  else if (ndim() == 3) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, ebx2_r);
  }
  else {
    throw std::runtime_error("Lorenzo only works for 123-D.");
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}


template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num, T* eb_list, uint16_t const radius,
    float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (data_len3.z == 1 and data_len3.y == 1)
      return 1;
    else if (data_len3.z == 1 and data_len3.y != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = dim3(256, 1, 1),
                 Tile2D = dim3(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1),
                 Tile3D = dim3(8, 8, 8);
  constexpr auto Block1D = dim3(256, 1, 1),
                 Block2D = dim3(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1),
                 Block3D = dim3(8, 8, 8);

  auto Grid1D = divide3(data_len3, Tile1D),
       Grid2D = divide3(data_len3, Tile2D),
       Grid3D = divide3(data_len3, Tile3D);

  // error bound
  //auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (ndim() == 1) {
    /*
    psz::KERNEL_CUHIP_prototype_c_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, ebx2_r);
    */
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_c_lorenzo_2d1l__eb_list<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, eb_list);
  }
  else if (ndim() == 3) {
    /*
    psz::KERNEL_CUHIP_prototype_c_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
            out_ot_num, radius, ebx2_r);
    */
  }
  else {
    throw std::runtime_error("Lorenzo only works for 123-D.");
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t, typename EqEb = uint8_t>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__tile_eb(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    const EqEb* tile_eb, T threshold, uint16_t const radius,
    float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  constexpr auto Tile2D = dim3(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1);
  constexpr auto Block2D = dim3(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1);
  auto Grid2D = divide3(data_len3, Tile2D);
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);
  psz::KERNEL_CUHIP_prototype_c_lorenzo_2d1l__tile_eb<T, LORENZO_TILE_DIM, Eq, EqEb>
      <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
          in_data, data_len3, data_leap3, out_eq, out_ot_val, out_ot_idx,
          out_ot_num, radius, tile_eb, threshold);
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}



// 1D row-wise Lorenzo: all rows parallel, pred = left neighbor only.
template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_c_lorenzo_row1d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* eb_list, uint16_t const /*radius*/,
    float* time_elapsed, void* stream)
{
  // Use a smaller effective radius (512) to guarantee HF code depth <= ~19 bits.
  // With bklen=8192 but only ~1024 distinct codes used, depth is well under 27.
  constexpr uint16_t EFF_RADIUS = 512;
  constexpr int ROWS_PER_BLOCK = 32;
  size_t r1 = data_len3.y;
  dim3 grid((r1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
  dim3 block(ROWS_PER_BLOCK, 1, 1);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_c_lorenzo_row1d__eb_list<T, Eq>
      <<<grid, block, 0, (cudaStream_t)stream>>>(
          in_data, data_len3, out_eq, out_ot_val, out_ot_idx, out_ot_num,
          EFF_RADIUS, eb_list);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

// Chunk-based 1D Lorenzo: each chunk processed by one thread in parallel.
// K=ceil(r2/CHUNK_SIZE) chunks per row → K×r1 parallel threads.
// Chunk boundaries stored as exact outliers (negligible overhead).
template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_c_lorenzo_chunk1d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* eb_list, uint16_t const /*radius*/,
    float* time_elapsed, void* stream)
{
  constexpr uint16_t EFF_RADIUS = 512;
  constexpr uint32_t CHUNK_SIZE = 10000;   // sequential steps per thread
  constexpr int THREADS_PER_BLOCK = 256;
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t K = (r2 + CHUNK_SIZE - 1) / CHUNK_SIZE;
  size_t total_chunks = r1 * K;
  dim3 grid((total_chunks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
  dim3 block(THREADS_PER_BLOCK, 1, 1);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_c_lorenzo_chunk1d__eb_list<T, Eq>
      <<<grid, block, 0, (cudaStream_t)stream>>>(
          in_data, data_len3, out_eq, out_ot_val, out_ot_idx, out_ot_num,
          EFF_RADIUS, eb_list, CHUNK_SIZE);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

// Correct variable-EB 2D Lorenzo: predicts in float space, quantizes residual
// with each element's own EB. Sequential scan (one thread) to maintain a
// consistent decompressed-value buffer across tile boundaries.
template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_correct_c_lorenzo_2d__eb_list(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,
    T* decomp_buf, T* eb_list, uint16_t const radius,
    float* time_elapsed, void* stream)
{
  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_correct_c_lorenzo_2d__eb_list<T, Eq>
      <<<1, 1, 0, (cudaStream_t)stream>>>(
          in_data, data_len3, out_eq, out_ot_val, out_ot_idx, out_ot_num,
          decomp_buf, radius, eb_list);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

////////////////////////////////////////////////////////////////////////////////
#define INSTANTIATIE_GPU_LORENZO_PROTO_C_2params(T, Eq)                     \
  template pszerror psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier<T, Eq>( \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,   \
      void* out_outlier, double const eb, uint16_t const radius,        \
      float* time_elapsed, void* stream);\
  template pszerror psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct<T, Eq>( \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,   \
      T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num, \
      double const eb, uint16_t const radius,        \
      float* time_elapsed, void* stream); \
  template pszerror psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, Eq>( \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,   \
      T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num, \
      T* eb_list, uint16_t const radius,        \
      float* time_elapsed, void* stream);

#define INSTANTIATIE_CHUNK1D_LORENZO_C_2params(T, Eq)                              \
  template pszerror psz::cuhip::GPU_PROTO_c_lorenzo_chunk1d__eb_list<T, Eq>(       \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,                   \
      T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,                  \
      T* eb_list, uint16_t const, float* time_elapsed, void* stream);

#define INSTANTIATIE_ROW1D_LORENZO_C_2params(T, Eq)                               \
  template pszerror psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq>(        \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,                   \
      T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,                  \
      T* eb_list, uint16_t const radius,                                           \
      float* time_elapsed, void* stream);

#define INSTANTIATIE_TILE_EB_LORENZO_C_3params(T, Eq, EqEb)                       \
  template pszerror psz::cuhip::                                                  \
      GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__tile_eb<       \
          T, Eq, EqEb>(                                                           \
          T* const in_data, dim3 const data_len3, Eq* const out_eq,               \
          T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,              \
          const EqEb* tile_eb, T threshold, uint16_t const radius,                \
          float* time_elapsed, void* stream);

#define INSTANTIATIE_CORRECT_LORENZO_C_2params(T, Eq)                              \
  template pszerror psz::cuhip::GPU_PROTO_correct_c_lorenzo_2d__eb_list<T, Eq>(   \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,                   \
      T* out_ot_val, uint32_t* out_ot_idx, uint32_t* out_ot_num,                  \
      T* decomp_buf, T* eb_list, uint16_t const radius,                            \
      float* time_elapsed, void* stream);

#define INSTANTIATIE_LORENZO_PROTO_C_1param(T) \
  INSTANTIATIE_GPU_LORENZO_PROTO_C_2params(T, uint16_t); \
  INSTANTIATIE_GPU_LORENZO_PROTO_C_2params(T, uint8_t); \
  INSTANTIATIE_CORRECT_LORENZO_C_2params(T, uint16_t); \
  INSTANTIATIE_ROW1D_LORENZO_C_2params(T, uint16_t); \
  INSTANTIATIE_CHUNK1D_LORENZO_C_2params(T, uint16_t); \
  INSTANTIATIE_TILE_EB_LORENZO_C_3params(T, uint16_t, uint8_t);
