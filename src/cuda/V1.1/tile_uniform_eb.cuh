#pragma once

#include <cstdint>

#include "lorenzo_tile_dim.h"

// Reduce one error-bound ID per tile and broadcast it back to every valid
// element. A common quantization unit enables the tile-prefix Lorenzo kernels.
template <typename T, typename Eq>
__global__ void kernel_uniformize_tile_eb(
    T* __restrict__ eb,
    Eq* __restrict__ eq_eb,
    int r1,
    int r2,
    T threshold)
{
    constexpr int tile_elements = LORENZO_TILE_DIM * LORENZO_TILE_DIM;
    __shared__ unsigned int tile_ids[tile_elements];

    const int x = blockIdx.x * LORENZO_TILE_DIM + threadIdx.x;
    const int y = blockIdx.y * LORENZO_TILE_DIM + threadIdx.y;
    const int tid = threadIdx.y * LORENZO_TILE_DIM + threadIdx.x;
    const bool in_bounds = x < r2 && y < r1;
    const size_t i = in_bounds ? static_cast<size_t>(y) * r2 + x : 0;

    tile_ids[tid] = in_bounds
        ? static_cast<unsigned int>(eq_eb[i])
        : 0xffffffffu;
    __syncthreads();

    for (int stride = tile_elements / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tile_ids[tid + stride] < tile_ids[tid]) {
            tile_ids[tid] = tile_ids[tid + stride];
        }
        __syncthreads();
    }

    if (in_bounds) {
        const Eq tile_id = static_cast<Eq>(tile_ids[0]);
        eq_eb[i] = tile_id;
        eb[i] = static_cast<T>(1ULL << (2 * static_cast<unsigned int>(tile_id)))
            * threshold;
    }
}
