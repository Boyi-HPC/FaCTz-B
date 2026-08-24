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

// easy algorithmic description
namespace psz {

template <
    typename T, int TileDim = 256, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_1d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  if (id < data_len3.x)
    data(0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;  // fuse
  else
    data(0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d)
      n = data(-d);  // like __shfl_up_sync(0x1f, var, d); warp_sync
    __syncthreads();
    if (t().x >= d) data(0) += n;
    __syncthreads();
  }

  if (id < data_len3.x) { out_data[id] = data(0) * ebx2; }
}

template <
    typename T, int TileDim = 16, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_2d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];

  auto id = gid2();
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  if (check_boundary2())
    data(0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;  // fuse
  else
    data(0, 0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d) n = data(-d, 0);
    __syncthreads();
    if (t().x >= d) data(0, 0) += n;
    __syncthreads();
  }

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().y >= d) n = data(0, -d);
    __syncthreads();
    if (t().y >= d) data(0, 0) += n;
    __syncthreads();
  }

  if (check_boundary2()) { out_data[id] = data(0, 0) * ebx2; }
}



template <
    typename T, int TileDim = LORENZO_TILE_DIM, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_2d1l__eb_list(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    T* eb_list)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];

  auto id = gid2();
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  if (check_boundary2())
    data(0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;  // fuse
  else
    data(0, 0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d) n = data(-d, 0);
    __syncthreads();
    if (t().x >= d) data(0, 0) += n;
    __syncthreads();
  }

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().y >= d) n = data(0, -d);
    __syncthreads();
    if (t().y >= d) data(0, 0) += n;
    __syncthreads();
  }

  if (check_boundary2()) { out_data[id] = data(0, 0) * (eb_list[id] * 2); }
}

template <
    typename T, int TileDim = LORENZO_TILE_DIM, typename Eq = uint16_t,
    typename EqEb = uint8_t>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_2d1l__tile_eb(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    const EqEb* tile_eb, T threshold)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];
  __shared__ T ebx2;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t tile_id = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int eb_id = (unsigned int)tile_eb[tile_id];
    ebx2 = ((T)(1ULL << (2 * eb_id)) * threshold) * (T)2;
  }

  auto id = gid2();
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };
  if (check_boundary2())
    data(0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;
  else
    data(0, 0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d) n = data(-d, 0);
    __syncthreads();
    if (t().x >= d) data(0, 0) += n;
    __syncthreads();
  }
  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().y >= d) n = data(0, -d);
    __syncthreads();
    if (t().y >= d) data(0, 0) += n;
    __syncthreads();
  }
  if (check_boundary2()) out_data[id] = data(0, 0) * ebx2;
}


template <typename T, int TileDim = 8, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_3d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim][TileDim + 1];

  auto id = gid3();
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  if (check_boundary3())
    data(0, 0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;
  else
    data(0, 0, 0) = 0;
  __syncthreads();

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().x >= dist) addend = data(-dist, 0, 0);
    __syncthreads();
    if (t().x >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().y >= dist) addend = data(0, -dist, 0);
    __syncthreads();
    if (t().y >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().z >= dist) addend = data(0, 0, -dist);
    __syncthreads();
    if (t().z >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  if (check_boundary3()) { out_data[id] = data(0, 0, 0) * ebx2; }
}

// Chunk-based 1D Lorenzo decompressor.
// Chunk boundaries are pre-scattered as exact values before this kernel.
// Each thread processes one chunk independently (all chunks parallel).
template <typename T, typename Eq = uint16_t>
__global__ void KERNEL_x_lorenzo_chunk1d__eb_list(
    Eq* const in_eq, T* const out_data,
    dim3 const data_len3, uint16_t const radius, T* eb_list,
    uint32_t chunk_size)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t K = (r2 + chunk_size - 1) / chunk_size;
  size_t chunk_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_id >= r1 * K) return;

  size_t row   = chunk_id / K;
  size_t chunk = chunk_id % K;
  size_t j0    = chunk * chunk_size;
  size_t j_end = (j0 + chunk_size < r2) ? j0 + chunk_size : r2;

  // Chunk boundary: already pre-scattered with exact value; use as predictor start.
  double prev = 0.0;
  if (chunk > 0) {
    size_t bid = row * r2 + j0;
    prev = (double)out_data[bid];  // exact value from pre-scatter
    j0++;
  }

  for (size_t j = j0; j < j_end; j++) {
    size_t id = row * r2 + j;
    Eq qi = in_eq[id];
    if (qi == 0) {
      prev = (double)out_data[id];  // outlier: pre-scattered exact value
    } else {
      double decomp = prev + (double)(qi - radius) * 2.0 * (double)eb_list[id];
      out_data[id] = (T)decomp;
      prev = decomp;
    }
  }
}

// 1D row-wise Lorenzo decompressor: d[j] = d[j-1] + (qi-R)*2*eb.
// All rows run in parallel; outliers pre-scattered before calling.
template <typename T, typename Eq = uint16_t>
__global__ void KERNEL_x_lorenzo_row1d__eb_list(
    Eq* const in_eq, T* const out_data,
    dim3 const data_len3, uint16_t const radius, T* eb_list)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= r1) return;

  double prev = 0.0;

  for (size_t j = 0; j < r2; j++) {
    size_t id = row * r2 + j;
    Eq qi = in_eq[id];
    if (qi == 0) {
      // Outlier: out_data[id] has exact original value from pre-scatter
      prev = (double)out_data[id];
    } else {
      double decomp = prev + (double)(qi - radius) * 2.0 * (double)eb_list[id];
      out_data[id] = (T)decomp;
      prev = decomp;
    }
  }
}

// Correct variable-EB 2D Lorenzo decompressor: sequential scan in float space.
// out_data must be zero-initialized before call.
// ot_val: outlier float residuals (cur - pred) in compression scan order.
// Compression is sequential so outliers appear in row-major order.
template <typename T, typename Eq = uint16_t>
__global__ void KERNEL_correct_x_lorenzo_2d__eb_list(
    Eq* const in_eq, T* const ot_val, T* const out_data,
    dim3 const data_len3, uint16_t const radius, T* eb_list)
{
  size_t r1 = data_len3.y, r2 = data_len3.x;
  uint32_t ot_ptr = 0;
  for (size_t i = 0; i < r1; i++) {
    for (size_t j = 0; j < r2; j++) {
      size_t id = i * r2 + j;
      double left    = (j > 0)           ? (double)out_data[id - 1]      : 0.0;
      double top     = (i > 0)           ? (double)out_data[id - r2]     : 0.0;
      double topleft = (i > 0 && j > 0) ? (double)out_data[id - r2 - 1] : 0.0;
      double pred = left + top - topleft;

      Eq qi = in_eq[id];
      if (qi == 0) {
        // Outlier: out_data[id] was pre-scattered with the exact original value.
        // Keep it as-is; no pred arithmetic needed.
        // (out_data[id] is already the correct value from pre-scatter)
      } else {
        out_data[id] = (T)(pred + (double)(qi - radius) * 2.0 * (double)eb_list[id]);
      }
    }
  }
}

}  // namespace psz

namespace psz::cuhip {

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    double const eb, int const radius, float* time_elapsed, void* stream)
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
    psz::KERNEL_CUHIP_prototype_x_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }
  else if (ndim() == 3) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}


template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_x_lorenzo_nd__eb_list(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream)
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
//
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l__eb_list<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, eb_list);
  }
  else if (ndim() == 3) {
//
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t, typename EqEb = uint8_t>
pszerror GPU_PROTO_x_lorenzo_nd__tile_eb(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    const EqEb* tile_eb, int tile_dim, T threshold, int const radius,
    float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);
  if (tile_dim == 8) {
    auto grid = divide3(data_len3, dim3(8, 8, 1));
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l__tile_eb<T, 8, Eq, EqEb>
        <<<grid, dim3(8, 8, 1), 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3,
            (uint16_t)radius, tile_eb, threshold);
  }
  else if (tile_dim == 16) {
    auto grid = divide3(data_len3, dim3(16, 16, 1));
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l__tile_eb<T, 16, Eq, EqEb>
        <<<grid, dim3(16, 16, 1), 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3,
            (uint16_t)radius, tile_eb, threshold);
  }
  else if (tile_dim == 32) {
    auto grid = divide3(data_len3, dim3(32, 32, 1));
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l__tile_eb<T, 32, Eq, EqEb>
        <<<grid, dim3(32, 32, 1), 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3,
            (uint16_t)radius, tile_eb, threshold);
  }
  else {
    DESTROY_GPUEVENT_PAIR;
    return CUSZ_FAIL_ONDISK_FILE_ERROR;
  }
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_x_lorenzo_row1d__eb_list(
    Eq* in_eq, T* out_data, dim3 const data_len3,
    T* eb_list, int const /*radius*/, float* time_elapsed, void* stream)
{
  constexpr uint16_t EFF_RADIUS = 512;  // must match compressor
  constexpr int ROWS_PER_BLOCK = 32;
  size_t r1 = data_len3.y;
  dim3 grid((r1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
  dim3 block(ROWS_PER_BLOCK, 1, 1);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_x_lorenzo_row1d__eb_list<T, Eq>
      <<<grid, block, 0, (cudaStream_t)stream>>>(
          in_eq, out_data, data_len3, EFF_RADIUS, eb_list);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_x_lorenzo_chunk1d__eb_list(
    Eq* in_eq, T* out_data, dim3 const data_len3,
    T* eb_list, int const /*radius*/, float* time_elapsed, void* stream)
{
  constexpr uint16_t EFF_RADIUS = 512;
  constexpr uint32_t CHUNK_SIZE = 10000;
  constexpr int THREADS_PER_BLOCK = 256;
  size_t r1 = data_len3.y, r2 = data_len3.x;
  size_t K = (r2 + CHUNK_SIZE - 1) / CHUNK_SIZE;
  size_t total_chunks = r1 * K;
  dim3 grid((total_chunks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
  dim3 block(THREADS_PER_BLOCK, 1, 1);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_x_lorenzo_chunk1d__eb_list<T, Eq>
      <<<grid, block, 0, (cudaStream_t)stream>>>(
          in_eq, out_data, data_len3, EFF_RADIUS, eb_list, CHUNK_SIZE);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_correct_x_lorenzo_2d__eb_list(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    T* eb_list, int const radius, float* time_elapsed, void* stream)
{
  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_correct_x_lorenzo_2d__eb_list<T, Eq>
      <<<1, 1, 0, (cudaStream_t)stream>>>(
          in_eq, in_outlier, out_data, data_len3, (uint16_t)radius, eb_list);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;
  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

////////////////////////////////////////////////////////////////////////////////
#define INSTANTIATIE_GPU_LORENZO_PROTO_X_2params(T, Eq)                         \
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_nd<T, Eq>(                  \
      Eq * in_eq, T * in_outlier, T * out_data, dim3 const data_len3, \
      double const eb, int const radius, float* time_elapsed, void* stream);\
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq>(         \
      Eq * in_eq, T * in_outlier, T * out_data, dim3 const data_len3, \
      T* eb_list, int const radius, float* time_elapsed, void* stream);

#define INSTANTIATIE_ROW1D_LORENZO_X_2params(T, Eq)                               \
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq>(        \
      Eq * in_eq, T * out_data, dim3 const data_len3,                             \
      T* eb_list, int const radius, float* time_elapsed, void* stream);

#define INSTANTIATIE_TILE_EB_LORENZO_X_3params(T, Eq, EqEb)                       \
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_nd__tile_eb<T, Eq, EqEb>(    \
      Eq * in_eq, T * in_outlier, T * out_data, dim3 const data_len3,            \
      const EqEb* tile_eb, int tile_dim, T threshold, int const radius,           \
      float* time_elapsed, void* stream);

#define INSTANTIATIE_CORRECT_LORENZO_X_2params(T, Eq)                             \
  template pszerror psz::cuhip::GPU_PROTO_correct_x_lorenzo_2d__eb_list<T, Eq>(   \
      Eq * in_eq, T * in_outlier, T * out_data, dim3 const data_len3,             \
      T* eb_list, int const radius, float* time_elapsed, void* stream);

#define INSTANTIATIE_CHUNK1D_LORENZO_X_2params(T, Eq)                             \
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_chunk1d__eb_list<T, Eq>(      \
      Eq * in_eq, T * out_data, dim3 const data_len3,                             \
      T* eb_list, int const, float* time_elapsed, void* stream);

#define INSTANTIATIE_LORENZO_PROTO_X_1param(T) \
  INSTANTIATIE_GPU_LORENZO_PROTO_X_2params(T, uint16_t); \
  INSTANTIATIE_GPU_LORENZO_PROTO_X_2params(T, uint8_t); \
  INSTANTIATIE_CORRECT_LORENZO_X_2params(T, uint16_t); \
  INSTANTIATIE_ROW1D_LORENZO_X_2params(T, uint16_t); \
  INSTANTIATIE_CHUNK1D_LORENZO_X_2params(T, uint16_t); \
  INSTANTIATIE_TILE_EB_LORENZO_X_3params(T, uint16_t, uint8_t);
