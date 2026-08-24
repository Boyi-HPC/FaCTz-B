#pragma once

#include <cstddef>
#include <cstdint>

template <typename T, typename Eq>
__global__ void kernel_fused_ebzero(
    const T* __restrict__ data_U,
    const T* __restrict__ data_V,
    T* __restrict__ eb_U,
    T* __restrict__ eb_V,
    Eq* __restrict__ eq_eb_U,
    Eq* __restrict__ eq_eb_V,
    uint32_t* __restrict__ zero_U_flags,
    uint32_t* __restrict__ zero_U_mask,
    uint32_t* __restrict__ zero_U_count,
    uint32_t* __restrict__ zero_V_flags,
    uint32_t* __restrict__ zero_V_mask,
    uint32_t* __restrict__ zero_V_count,
    T replacement_eb,
    T threshold,
    Eq replacement_id,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const bool zero_u = eq_eb_U[i] == Eq{0};
    const bool zero_v = eq_eb_V[i] == Eq{0};
    zero_U_flags[i] = zero_u;
    zero_V_flags[i] = zero_v;

    if (zero_u) {
        atomicOr(zero_U_mask + (i >> 5), 1u << (i & 31u));
        atomicAdd(zero_U_count, 1u);
        eq_eb_U[i] = replacement_id;
    }
    if (eb_U[i] <= threshold) eb_U[i] = replacement_eb;

    if (zero_v) {
        atomicOr(zero_V_mask + (i >> 5), 1u << (i & 31u));
        atomicAdd(zero_V_count, 1u);
        eq_eb_V[i] = replacement_id;
    }
    if (eb_V[i] <= threshold) eb_V[i] = replacement_eb;
}
