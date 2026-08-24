#pragma once

#include <cstddef>
#include <cstdint>

// Compact zero-EB values and indices while replacing their temporary error
// bounds. This preserves the naive pipeline's semantics in one array pass.
template <typename T, typename Eq>
__global__ void kernel_fused_ebzero(
    const T* __restrict__ data_U,
    const T* __restrict__ data_V,
    T* __restrict__ eb_U,
    T* __restrict__ eb_V,
    Eq* __restrict__ eq_eb_U,
    Eq* __restrict__ eq_eb_V,
    T* __restrict__ zero_U_values,
    uint32_t* __restrict__ zero_U_indices,
    uint32_t* __restrict__ zero_U_count,
    T* __restrict__ zero_V_values,
    uint32_t* __restrict__ zero_V_indices,
    uint32_t* __restrict__ zero_V_count,
    T replacement_eb,
    T threshold,
    Eq replacement_id,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (eq_eb_U[i] == Eq{0}) {
        const uint32_t out = atomicAdd(zero_U_count, 1u);
        zero_U_values[out] = data_U[i];
        zero_U_indices[out] = static_cast<uint32_t>(i);
        eq_eb_U[i] = replacement_id;
    }
    if (eb_U[i] <= threshold) eb_U[i] = replacement_eb;

    if (eq_eb_V[i] == Eq{0}) {
        const uint32_t out = atomicAdd(zero_V_count, 1u);
        zero_V_values[out] = data_V[i];
        zero_V_indices[out] = static_cast<uint32_t>(i);
        eq_eb_V[i] = replacement_id;
    }
    if (eb_V[i] <= threshold) eb_V[i] = replacement_eb;
}
