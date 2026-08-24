#pragma once
#include <stdint.h>
#include "lorenzo_tile_dim.h"

// Reduce per-element EB exponent IDs to one minimum ID per Lorenzo tile. The
// tile Lorenzo kernels consume this compact array directly, so no full-size
// float EB array or per-element uniformized EB array is needed.
template<typename Eq2>
__global__ void kernel_reduce_tile_eb_pair(
    const Eq2* __restrict__ eq_dEb_U,
    const Eq2* __restrict__ eq_dEb_V,
    Eq2* __restrict__ tile_eq_dEb_U,
    Eq2* __restrict__ tile_eq_dEb_V,
    int r1, int r2)
{
    __shared__ unsigned int warp_min_U[32];
    __shared__ unsigned int warp_min_V[32];

    int x = blockIdx.x * LORENZO_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * LORENZO_TILE_DIM + threadIdx.y;
    int tid = threadIdx.y * LORENZO_TILE_DIM + threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int num_warps = (LORENZO_TILE_DIM * LORENZO_TILE_DIM + 31) / 32;
    bool in = (x < r2 && y < r1);
    size_t id = (size_t)y * r2 + x;

    unsigned int min_U = in ? (unsigned int)eq_dEb_U[id] : 0xffffffffu;
    unsigned int min_V = in ? (unsigned int)eq_dEb_V[id] : 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        min_U = min(min_U, __shfl_down_sync(0xffffffffu, min_U, offset));
        min_V = min(min_V, __shfl_down_sync(0xffffffffu, min_V, offset));
    }
    if (lane == 0) {
        warp_min_U[warp] = min_U;
        warp_min_V[warp] = min_V;
    }
    __syncthreads();

    if (warp == 0) {
        min_U = lane < num_warps ? warp_min_U[lane] : 0xffffffffu;
        min_V = lane < num_warps ? warp_min_V[lane] : 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            min_U = min(min_U, __shfl_down_sync(0xffffffffu, min_U, offset));
            min_V = min(min_V, __shfl_down_sync(0xffffffffu, min_V, offset));
        }
        if (lane == 0) {
            size_t tile_id = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
            tile_eq_dEb_U[tile_id] = (Eq2)min_U;
            tile_eq_dEb_V[tile_id] = (Eq2)min_V;
        }
    }
}
