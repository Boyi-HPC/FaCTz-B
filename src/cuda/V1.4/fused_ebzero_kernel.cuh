#pragma once

#include <stddef.h>
#include <stdint.h>

#include "lorenzo_tile_dim.h"

// OR one tile-row ballot into the linear row-major bitmask. A row can straddle
// two words when r2 is not divisible by 32, so adjacent tiles may share words.
__device__ __forceinline__ void atomic_or_linear_ballot(
    uint32_t* mask, size_t mask_words, size_t linear_start, uint32_t bits)
{
    if (bits == 0) return;

    const size_t word = linear_start >> 5;
    const unsigned int shift = (unsigned int)(linear_start & 31u);
    atomicOr(mask + word, bits << shift);
    if (shift != 0) {
        const uint32_t high = bits >> (32u - shift);
        if (high != 0 && word + 1 < mask_words) atomicOr(mask + word + 1, high);
    }
}

// One block owns one Lorenzo tile. Eight warps walk the tile rows and jointly:
//   * detect land and emit its linear bitmask;
//   * apply land replacement before classifying zero-EB values;
//   * emit U/V zero-EB bitmasks and replace zero IDs;
//   * reduce the corrected IDs to one minimum per tile.
// The 32x8 launch supports every configured tile dimension up to 32.
template<typename T, typename Eq2>
__global__ void kernel_classify_specials_and_tile_eb(
    const T* __restrict__ dU, const T* __restrict__ dV,
    Eq2* __restrict__ eq_dEb_U, Eq2* __restrict__ eq_dEb_V,
    uint32_t* __restrict__ zero_eb_U_mask,
    uint32_t* __restrict__ zero_eb_V_mask,
    uint32_t* __restrict__ land_mask,
    Eq2* __restrict__ tile_eq_dEb_U,
    Eq2* __restrict__ tile_eq_dEb_V,
    Eq2 replace_id, size_t r1, size_t r2, size_t mask_words,
    bool replace_land, bool replace_zero_eb)
{
    static_assert(LORENZO_TILE_DIM <= 32, "LORENZO_TILE_DIM must be <= 32");

    constexpr unsigned int full_mask = 0xffffffffu;
    constexpr int num_worker_warps = 8;
    __shared__ unsigned int warp_min_U[num_worker_warps];
    __shared__ unsigned int warp_min_V[num_worker_warps];

    const int lane = threadIdx.x;
    const int worker_warp = threadIdx.y;
    const size_t tile_x0 = (size_t)blockIdx.x * LORENZO_TILE_DIM;
    const size_t tile_y0 = (size_t)blockIdx.y * LORENZO_TILE_DIM;
    unsigned int local_min_U = 0xffffffffu;
    unsigned int local_min_V = 0xffffffffu;

    for (int local_y = worker_warp; local_y < LORENZO_TILE_DIM;
         local_y += num_worker_warps) {
        const size_t y = tile_y0 + (size_t)local_y;
        const size_t x = tile_x0 + (size_t)lane;
        const bool in = lane < LORENZO_TILE_DIM && y < r1 && x < r2;
        const size_t i = in ? y * r2 + x : 0;

        bool is_land = false;
        bool zero_U = false;
        bool zero_V = false;
        unsigned int eb_U = 0xffffffffu;
        unsigned int eb_V = 0xffffffffu;

        if (in) {
            is_land = dU[i] == (T)0 && dV[i] == (T)0;
            Eq2 eu = eq_dEb_U[i];
            Eq2 ev = eq_dEb_V[i];

            if (replace_land && is_land) eu = ev = replace_id;
            zero_U = replace_zero_eb && eu == (Eq2)0;
            zero_V = replace_zero_eb && ev == (Eq2)0;
            if (zero_U) eu = replace_id;
            if (zero_V) ev = replace_id;

            eq_dEb_U[i] = eu;
            eq_dEb_V[i] = ev;
            eb_U = (unsigned int)eu;
            eb_V = (unsigned int)ev;
        }

        const uint32_t land_bits = __ballot_sync(full_mask, is_land);
        const uint32_t zero_U_bits = __ballot_sync(full_mask, zero_U);
        const uint32_t zero_V_bits = __ballot_sync(full_mask, zero_V);
        if (lane == 0 && y < r1) {
            const size_t row_start = y * r2 + tile_x0;
            atomic_or_linear_ballot(land_mask, mask_words, row_start, land_bits);
            atomic_or_linear_ballot(zero_eb_U_mask, mask_words, row_start, zero_U_bits);
            atomic_or_linear_ballot(zero_eb_V_mask, mask_words, row_start, zero_V_bits);
        }

        local_min_U = min(local_min_U, eb_U);
        local_min_V = min(local_min_V, eb_V);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min_U = min(
            local_min_U, __shfl_down_sync(full_mask, local_min_U, offset));
        local_min_V = min(
            local_min_V, __shfl_down_sync(full_mask, local_min_V, offset));
    }
    if (lane == 0) {
        warp_min_U[worker_warp] = local_min_U;
        warp_min_V[worker_warp] = local_min_V;
    }
    __syncthreads();

    if (worker_warp == 0) {
        unsigned int min_U = lane < num_worker_warps
            ? warp_min_U[lane] : 0xffffffffu;
        unsigned int min_V = lane < num_worker_warps
            ? warp_min_V[lane] : 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            min_U = min(min_U, __shfl_down_sync(full_mask, min_U, offset));
            min_V = min(min_V, __shfl_down_sync(full_mask, min_V, offset));
        }
        if (lane == 0 && tile_eq_dEb_U && tile_eq_dEb_V) {
            const size_t tile_id = (size_t)blockIdx.y * gridDim.x + blockIdx.x;
            tile_eq_dEb_U[tile_id] = (Eq2)min_U;
            tile_eq_dEb_V[tile_id] = (Eq2)min_V;
        }
    }
}

// Convert compact bitmask words to per-word counts. The count arrays are then
// scanned in place. Land only needs a total, accumulated once per warp here.
__global__ void kernel_mask_word_counts_pair_and_land(
    const uint32_t* __restrict__ zero_eb_U_mask,
    const uint32_t* __restrict__ zero_eb_V_mask,
    const uint32_t* __restrict__ land_mask,
    uint32_t* __restrict__ zero_eb_U_word_offsets,
    uint32_t* __restrict__ zero_eb_V_word_offsets,
    uint32_t* __restrict__ land_count,
    size_t word_count)
{
    const size_t word = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_land_count = 0;
    if (word < word_count) {
        zero_eb_U_word_offsets[word] = __popc(zero_eb_U_mask[word]);
        zero_eb_V_word_offsets[word] = __popc(zero_eb_V_mask[word]);
        local_land_count = __popc(land_mask[word]);
    }

    const unsigned int full_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_land_count +=
            __shfl_down_sync(full_mask, local_land_count, offset);
    }
    if ((threadIdx.x & 31) == 0 && local_land_count != 0) {
        atomicAdd(land_count, local_land_count);
    }
}

__global__ void kernel_finalize_zeroeb_counts(
    const uint32_t* __restrict__ zero_eb_U_mask,
    const uint32_t* __restrict__ zero_eb_V_mask,
    const uint32_t* __restrict__ zero_eb_U_word_offsets,
    const uint32_t* __restrict__ zero_eb_V_word_offsets,
    uint32_t* __restrict__ zero_eb_U_count,
    uint32_t* __restrict__ zero_eb_V_count,
    size_t word_count)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (word_count == 0) {
            *zero_eb_U_count = 0;
            *zero_eb_V_count = 0;
            return;
        }
        const size_t last = word_count - 1;
        *zero_eb_U_count = zero_eb_U_word_offsets[last] + __popc(zero_eb_U_mask[last]);
        *zero_eb_V_count = zero_eb_V_word_offsets[last] + __popc(zero_eb_V_mask[last]);
    }
}

// One warp owns one mask word. popc of the lower set bits gives a stable rank,
// preserving the row-major zeroEB value order expected by the decompressor.
template<typename T>
__global__ void kernel_compact_zeroeb_values_pair_by_word(
    const T* __restrict__ dU, const T* __restrict__ dV,
    const uint32_t* __restrict__ zero_eb_U_mask,
    const uint32_t* __restrict__ zero_eb_V_mask,
    const uint32_t* __restrict__ zero_eb_U_word_offsets,
    const uint32_t* __restrict__ zero_eb_V_word_offsets,
    T* __restrict__ zero_eb_U_values,
    T* __restrict__ zero_eb_V_values,
    size_t n, size_t word_count)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const size_t word = (size_t)blockIdx.x * warps_per_block + warp;
    if (word >= word_count) return;

    const size_t i = (word << 5) + (size_t)lane;
    const uint32_t mask_U = zero_eb_U_mask[word];
    const uint32_t mask_V = zero_eb_V_mask[word];
    const uint32_t bit = 1u << lane;
    const uint32_t lower_bits = lane == 0 ? 0u : bit - 1u;

    if (i < n && (mask_U & bit)) {
        const uint32_t rank = __popc(mask_U & lower_bits);
        zero_eb_U_values[zero_eb_U_word_offsets[word] + rank] = dU[i];
    }
    if (i < n && (mask_V & bit)) {
        const uint32_t rank = __popc(mask_V & lower_bits);
        zero_eb_V_values[zero_eb_V_word_offsets[word] + rank] = dV[i];
    }
}
