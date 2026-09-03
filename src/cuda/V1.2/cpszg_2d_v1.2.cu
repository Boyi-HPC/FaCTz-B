#include "sz_cp_preserve_utils.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz3_utils.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"
#include <chrono>
#include <limits>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include "kernel/lrz/lproto.hh"
#include "lorenzo_tile_dim.h"
#include "fused_ebzero_kernel.cuh"
#include "tile_uniform_eb.cuh"
#include "cucpsz_2d_format.hpp"

using namespace std;

#define BLOCKSIZE_X 32 //default32
#define BLOCKSIZE_Y 16 //default16
#define NUM_PRE_THREAD 4
#define RADIUS 512

template<typename T>
struct ReplaceZero {
    T replace_num;
    
    __host__ __device__
    ReplaceZero(T replace_num) : replace_num(replace_num) {}

    __host__ __device__
    T operator()(T x) const {
        return (x == 0) ? replace_num : x;
    }
};

template<typename T>
struct ReplaceLessThreshold {
    T replace_num;
    T threshold;

    __host__ __device__
    ReplaceLessThreshold(T replace_num, T threshold)
        : replace_num(replace_num), threshold(threshold) {}

    __host__ __device__
    T operator()(T x) const {
        return (x <= threshold) ? replace_num : x;
    }
};

template<typename T>
struct IsZero {
    __host__ __device__
    IsZero() {}

    __host__ __device__
    bool operator()(T x) const {
        return x == 0;
    }
};

template<typename T>
struct IsLessThreshold {
    T threshold;

    __host__ __device__
    IsLessThreshold(T threshold) : threshold(threshold) {}

    __host__ __device__
    bool operator()(T x) const {
        return x <= threshold;
    }
};



struct OtData {
    float* val;
    uint32_t* idx;
    uint32_t* num;
    uint32_t* h_num;
};

void allocOtData(OtData& data, size_t size) {
    cudaMalloc(&data.val, size * sizeof(float));
    cudaMalloc(&data.idx, size * sizeof(uint32_t));
    cudaMalloc(&data.num, sizeof(uint32_t));
    cudaMallocHost(&data.h_num, sizeof(uint32_t));
    cudaMemset(data.num, 0, sizeof(uint32_t));
    *data.h_num = 0;
}

void freeOtData(OtData& data) {
    cudaFree(data.val);
    cudaFree(data.idx);
    cudaFree(data.num);
    cudaFreeHost(data.h_num);
}


template<typename T>
[[nodiscard]] constexpr inline T max_eb_to_keep_sign_2d_offline_2(const T volatile u0, const T volatile u1, const int degree=2){
    T positive = 0;
    T negative = 0;
    accumulate_2d(u0, positive, negative);
    accumulate_2d(u1, positive, negative);
    return max_eb_to_keep_sign(positive, negative, degree);
}

template<typename T>
[[nodiscard]] constexpr inline T gpu_max_eb_to_keep_sign_2d_offline_2_degree2(const T u0, const T u1){
    //if(value >= 0) positive += value;
	// else negative += - value;
    T positive = (u0>=0 ? u0 : 0) + (u1>=0? u1 : 0);
    T negative = (u0<0 ? -u0 : 0) + (u1<0? -u1 : 0);
    T P = sqrt(positive);
    T N = sqrt(negative);
    return fabs(P - N)/(P + N);
}

template<typename T>
[[nodiscard]] constexpr inline T max_eb_to_keep_sign_2d_offline_4(const T volatile u0, const T volatile u1, const T volatile u2, const T volatile u3, const int degree=2){
    T positive = 0;
    T negative = 0;
    accumulate_2d(u0, positive, negative);
    accumulate_2d(u1, positive, negative);
    accumulate_2d(u2, positive, negative);
    accumulate_2d(u3, positive, negative);
    return max_eb_to_keep_sign(positive, negative, degree);
}

template<typename T>
[[nodiscard]] constexpr inline T gpu_max_eb_to_keep_sign_2d_offline_4_degree2(const T u0, const T u1, const T u2, const T u3){
    T positive = (u0>=0 ? u0 : 0) + (u1>=0? u1 : 0) + (u2>=0 ? u2 : 0) + (u3>=0? u3 : 0);
    T negative = (u0<0 ? -u0 : 0) + (u1<0? -u1 : 0) + (u2<0 ? -u2 : 0) + (u3<0? -u3 : 0);
    T P = sqrt(positive);
    T N = sqrt(negative);
    return fabs(P - N)/(P + N);
}


template<typename T>
[[nodiscard]] constexpr inline double max_eb_to_keep_position_and_type(const T volatile u0, const T volatile u1, const T volatile u2, const T volatile v0, const T volatile v1, const T volatile v2)
{	
    T u0v1 = u0 * v1;
    T u1v0 = u1 * v0;
    T u0v2 = u0 * v2;
    T u2v0 = u2 * v0;
    T u1v2 = u1 * v2;
    T u2v1 = u2 * v1;
    T det = u0v1 - u1v0 + u1v2 - u2v1 + u2v0 - u0v2;
    T eb = 0;
    if(det != 0){
        bool f1 = (det / (u2v0 - u0v2) >= 1);//u0v1 - u1v0 + u1v2 - u2v1>=0
        bool f2 = (det / (u1v2 - u2v1) >= 1);//u0v1 - u1v0 + u2v0 - u0v2>=0
        bool f3 = (det / (u0v1 - u1v0) >= 1);//u1v2 - u2v1 + u2v0 - u0v2>=0
        if(f1 && f2 && f3){
            eb=0;
        }
        else{
            // no critical point
            eb = 0;
            if(!f1){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u2v0, -u0v2), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u1v2, -u2v1));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u2, u0, v2, v0), max_eb_to_keep_sign_4(u0, u1, u2, v0, v1, v2));
                eb = MAX(eb, eb_cur);
            }
            if(!f2){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u1v2, -u2v1), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u2v0, -u0v2));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u1, u2, v1, v2), max_eb_to_keep_sign_4(u2, u0, u1, v2, v0, v1));
                eb = MAX(eb, eb_cur);
            }
            if(!f3){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u0v1, -u1v0), max_eb_to_keep_sign_2d_offline_4(u1v2, -u2v1, u2v0, -u0v2));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u0, u1, v0, v1), max_eb_to_keep_sign_4(u1, u2, u0, v1, v2, v0));
                eb = MAX(eb, eb_cur);
            }
            // eb = MINF(eb, DEFAULT_EB);
        }
    }
    return eb;
}

template<typename T>
[[nodiscard]] constexpr inline double gpu_max_eb_to_keep_position_and_type(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2)
{
    auto gpu_minf = [](auto a, auto b) -> T{ return (a<b)?a:b; };
#define U0V1 u0*v1
#define U1V0 u1*v0
#define U0V2 u0*v2
#define U2V0 u2*v0
#define U1V2 u1*v2
#define U2V1 u2*v1
    T det = U0V1 - U1V0 + U1V2 - U2V1 + U2V0 - U0V2;
    T eb = 0;
    if(det != 0)
    {   
        T d1 = U2V0 - U0V2;
        T d2 = U1V2 - U2V1;
        T d3 = U0V1 - U1V0;
        bool f1 = (det / d1 >= T(1));
        bool f2 = (det / d2 >= T(1));
        bool f3 = (det / d3 >= T(1)); 
        if(!f1){
            T pos1 = (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg1 = (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0) + (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0);
            T neg2 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0) + (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);

            eb = MAX(eb, eb_cur);
        }
        if(!f2){
            T pos1 = (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0);
            T neg1 = (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0) + (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg2 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0) + (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);
            
            eb = MAX(eb, eb_cur);
        }
        if(!f3){
            T pos1 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0);
            T neg1 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0) + (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg2 = (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0) + (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);

            eb = MAX(eb, eb_cur);
        }
        eb = gpu_minf(eb, 1);
    }
    return eb;
}

//version 2, single thread muti-compute 32*32 data map to 32*8
template <typename T, typename Eq2 = uint16_t, int TileDim_X = BLOCKSIZE_X, int TileDim_Y = BLOCKSIZE_Y>
__global__ void derive_eb_offline_v2(
        const T* __restrict__ dU, const T* __restrict__ dV,
        Eq2* __restrict eq_dEb_U, Eq2* __restrict eq_dEb_V,
        int r1, int r2, T max_pwr_eb){
    __shared__ T buf_U[TileDim_Y][TileDim_X+1];
    __shared__ T buf_V[TileDim_Y][TileDim_X+1];
    __shared__ T per_cell_eb_L[TileDim_Y][TileDim_X+1];
    __shared__ T per_cell_eb_U[TileDim_Y][TileDim_X+1];
    __shared__ T buf_eb[TileDim_Y][TileDim_X+1];  
    int row = blockIdx.y * (blockDim.y-2) + threadIdx.y; // global row index
    int col = blockIdx.x * (blockDim.x-2) + threadIdx.x; // global col index
    int localRow = threadIdx.y; // local row index
    int localCol = threadIdx.x; // local col index
    //#define localRow threadIdx.y
    //#define localCol threadIdx.x

    T localmin = max_pwr_eb;

    // load data from global memory to shared memory
    if(row < r1 && col < r2){
        buf_U[localRow][localCol] = dU[row * r2 + col];
        buf_V[localRow][localCol] = dV[row * r2 + col];
    }
    __syncthreads();
    
    if(localRow<TileDim_Y-1 && localCol<TileDim_X-1){
        per_cell_eb_U[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow][localCol+1], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow][localCol+1], buf_V[localRow+1][localCol+1]);
        per_cell_eb_L[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow+1][localCol], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow+1][localCol], buf_V[localRow+1][localCol+1]);
    }
    __syncthreads();

    if(localRow<TileDim_Y-2 && localCol<TileDim_X-2)
    {
        auto temp = per_cell_eb_U[localRow][localCol];
        localmin = min(localmin, temp);
        temp =  per_cell_eb_L[localRow][localCol];
        localmin = min(localmin, temp);
        temp =  per_cell_eb_U[localRow+1][localCol];
        localmin = min(localmin, temp);
        temp = per_cell_eb_L[localRow][localCol+1];
        localmin = min(localmin, temp);
        temp = per_cell_eb_U[localRow+1][localCol+1];
        localmin = min(localmin, temp);
        temp = per_cell_eb_L[localRow+1][localCol+1];
        localmin = min(localmin, temp);
        buf_eb[localRow][localCol] = localmin;
    }
    __syncthreads();

    if(row<r1-2 && col<r2-2 && localRow<TileDim_Y-2 && localCol<TileDim_X-2)
    {
        T threshold = (T)(1.0 / (1 << 20));
        int id;
        auto temp = buf_eb[localRow][localCol] * fabs(buf_U[localRow+1][localCol+1]);
        if(temp <= threshold){
            temp = 0;
            id = 0;
        }
        if(temp > threshold){
            id = log2(temp / threshold)/2.0;
            temp = (T)(1ULL << (2 * id)) * threshold;
        }

        eq_dEb_U[(row+1) * r2 + (col+1)] = id;

        temp = buf_eb[localRow][localCol] * fabs(buf_V[localRow+1][localCol+1]);
        if(temp <= threshold){
            temp = 0;
            id = 0;
        }
        if(temp > threshold){
            id = log2(temp / threshold)/2.0;
            temp = (T)(1ULL << (2 * id)) * threshold;
        }

        eq_dEb_V[(row+1) * r2 + (col+1)] = id;
    }
    __syncthreads();

    if((row == 0 || col ==0 || row==r1-1 || col == r2-1)&&(row<r1-1 && col<r2-1)){
        eq_dEb_U[row * r2 + col] = 0;
        eq_dEb_V[row * r2 + col] = 0;
    }
    __syncthreads();
}

template <typename T>
__global__ void kernel_compact_zeroeb_values(
    const T* __restrict__ data,
    const uint32_t* __restrict__ mask,
    const uint32_t* __restrict__ offsets,
    T* __restrict__ zeroeb_values,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((mask[i >> 5] >> (i & 31u)) & 1u) {
        zeroeb_values[offsets[i]] = data[i];
    }
}

template <typename Eq2>
static void copy_tile_eb_payloads(
    const Eq2* d_tile_U, const Eq2* d_tile_V,
    size_t tile_count,
    uint8_t*& h_tile_U, uint8_t*& h_tile_V,
    size_t& tile_bytes_U, size_t& tile_bytes_V)
{
    tile_bytes_U = tile_count * sizeof(Eq2);
    tile_bytes_V = tile_count * sizeof(Eq2);

    h_tile_U = new uint8_t[tile_bytes_U ? tile_bytes_U : 1];
    h_tile_V = new uint8_t[tile_bytes_V ? tile_bytes_V : 1];
    cudaMemcpy(h_tile_U, d_tile_U, tile_bytes_U, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tile_V, d_tile_V, tile_bytes_V, cudaMemcpyDeviceToHost);
}

// Header is naturally padded to 96 bytes on the current ABI.
// File layout after header:
//   [payload 0: HF eq_U]
//   [payload 1: HF eq_V]
//   [payload 2: HF eq_dEb_U for row1d, raw per-tile eq_dEb_U for tile variant]
//   [payload 3: HF eq_dEb_V for row1d, raw per-tile eq_dEb_V for tile variant]
//   [ot_idx_U: ot_count_U * 4 bytes]
//   [ot_val_U: ot_count_U * 4 bytes]
//   [zeroeb_mask_U: ceil(n/32) * 4 bytes if zeroeb_count_U > 0]
//   [zeroeb_val_U: zeroeb_count_U * 4 bytes]
//   [same 4 arrays for V, with zero-EB mask instead of zero-EB idx]
// Write the .cucpsz file.
// Per field: ot_idx+ot_val (outlier quant codes for pre-scatter),
//            zeroeb_mask+zeroeb_val (original floats, row-major compact order).
static double write_cucpsz(
    const char* fname,
    size_t r1, size_t r2, float max_pwr_eb,
    uint8_t* hf_blobs[4], size_t hf_lens[4],
    uint32_t ot_count_U,    const uint32_t* ot_idx_U,    const float* ot_val_U,
    uint32_t zeroeb_count_U,const uint8_t* zeroeb_mask_U,const float* zeroeb_val_U,
    uint32_t ot_count_V,    const uint32_t* ot_idx_V,    const float* ot_val_V,
    uint32_t zeroeb_count_V,const uint8_t* zeroeb_mask_V,const float* zeroeb_val_V,
    const uint8_t* land_bitpack, size_t land_bitpack_bytes,
    uint32_t lorenzo_variant, uint32_t lorenzo_tile_dim)
{
    CucpszHeader hdr{};
    memcpy(hdr.magic, "CUCPSZ\0\0", 8);
    hdr.r1 = r1; hdr.r2 = r2;
    hdr.max_pwr_eb = max_pwr_eb;
    hdr.ot_count_U = ot_count_U;   hdr.ot_count_V = ot_count_V;
    hdr.zeroeb_count_U = zeroeb_count_U; hdr.zeroeb_count_V = zeroeb_count_V;
    for (int i = 0; i < 4; i++) hdr.hf_blob_len[i] = hf_lens[i];
    hdr.land_bitpack_bytes = land_bitpack_bytes;
    hdr.lorenzo_variant = lorenzo_variant;
    hdr.lorenzo_tile_dim = lorenzo_tile_dim;

    using IoClock = std::chrono::steady_clock;
    double write_seconds = 0.0;
    auto io_start = IoClock::now();
    FILE* f = fopen(fname, "wb");
    write_seconds += std::chrono::duration<double>(IoClock::now() - io_start).count();
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        return -1.0;
    }
    bool write_ok = true;
    auto timed_fwrite = [&](const void* data, size_t item_size, size_t count) {
        if (count == 0) return;
        auto start = IoClock::now();
        size_t written = fwrite(data, item_size, count, f);
        write_seconds += std::chrono::duration<double>(IoClock::now() - start).count();
        if (written != count) write_ok = false;
    };
    timed_fwrite(&hdr, sizeof(hdr), 1);

    size_t zeroeb_mask_bytes_U = zeroeb_count_U ? bitmask_word_bytes(r1 * r2) : 0;
    size_t zeroeb_mask_bytes_V = zeroeb_count_V ? bitmask_word_bytes(r1 * r2) : 0;
    size_t total_bytes = sizeof(hdr) + land_bitpack_bytes;
    for (int i = 0; i < 4; i++) total_bytes += hf_lens[i];
    total_bytes += ot_count_U * (sizeof(uint32_t) + sizeof(float)) + zeroeb_mask_bytes_U + zeroeb_count_U * sizeof(float);
    total_bytes += ot_count_V * (sizeof(uint32_t) + sizeof(float)) + zeroeb_mask_bytes_V + zeroeb_count_V * sizeof(float);
    auto pct = [&](size_t bytes) { return bytes * 100.0 / total_bytes; };

    const char* hf_names[4] = {
        "eq_U",
        "eq_V",
        lorenzo_variant == 1u ? "tile_dEb_U" : "eq_dEb_U",
        lorenzo_variant == 1u ? "tile_dEb_V" : "eq_dEb_V"
    };
    for (int i = 0; i < 4; i++) {
        timed_fwrite(hf_blobs[i], 1, hf_lens[i]);
        printf("  payload %-10s : %zu bytes (%.1f%%)\n", hf_names[i], hf_lens[i], pct(hf_lens[i]));
    }

    // Land bitpack (1 bit per element, 1 = U=V=0)
    timed_fwrite(land_bitpack, 1, land_bitpack_bytes);
    printf("  land bitpack       : %zu bytes (%.1f%%)\n", land_bitpack_bytes, pct(land_bitpack_bytes));

    // U: outlier indices+values, then zeroeb mask+original_floats
    timed_fwrite(ot_idx_U, sizeof(uint32_t), ot_count_U);
    timed_fwrite(ot_val_U, sizeof(float), ot_count_U);
    timed_fwrite(zeroeb_mask_U, 1, zeroeb_mask_bytes_U);
    timed_fwrite(zeroeb_val_U, sizeof(float), zeroeb_count_U);
    size_t sp_U_bytes = ot_count_U * (sizeof(uint32_t) + sizeof(float)) +
                        zeroeb_mask_bytes_U + zeroeb_count_U * sizeof(float);
    printf("  special U (ot=%u idx/val + zeroeb=%u mask/val): %zu bytes (%.1f%%)\n",
           ot_count_U, zeroeb_count_U, sp_U_bytes, pct(sp_U_bytes));
    printf("    zeroeb U mask    : %zu bytes\n", zeroeb_mask_bytes_U);

    // V
    timed_fwrite(ot_idx_V, sizeof(uint32_t), ot_count_V);
    timed_fwrite(ot_val_V, sizeof(float), ot_count_V);
    timed_fwrite(zeroeb_mask_V, 1, zeroeb_mask_bytes_V);
    timed_fwrite(zeroeb_val_V, sizeof(float), zeroeb_count_V);
    size_t sp_V_bytes = ot_count_V * (sizeof(uint32_t) + sizeof(float)) +
                        zeroeb_mask_bytes_V + zeroeb_count_V * sizeof(float);
    printf("  special V (ot=%u idx/val + zeroeb=%u mask/val): %zu bytes (%.1f%%)\n",
           ot_count_V, zeroeb_count_V, sp_V_bytes, pct(sp_V_bytes));
    printf("    zeroeb V mask    : %zu bytes\n", zeroeb_mask_bytes_V);

    printf("  total              : %zu bytes\n", total_bytes);
    auto close_start = IoClock::now();
    if (fclose(f) != 0) write_ok = false;
    write_seconds += std::chrono::duration<double>(IoClock::now() - close_start).count();
    if (!write_ok) {
        fprintf(stderr, "Failed while writing %s\n", fname);
        return -1.0;
    }
    printf("Written %s\n", fname);
    return write_seconds;
}

struct CpCompressDebugOptions {
    bool run_derive_eb = true;
    bool verify_derive_eb = false;
    bool deal_with_land_data = true;
    bool deal_with_ebzero = true;
    bool run_cusz_lorenzo = false;
    bool test_cusz_lorenzo = true;
    bool run_quantization_lorenzo = true;
    bool run_hf = true;
    bool compute_ratio = true;
    bool overall_performance = false;
    bool write_debug_data = false;
    bool max_data = true;
    bool test_rightness = true;
    // When true: uniformize eb per 16x16 tile (min within tile) and use the
    // parallel prefix-sum tile Lorenzo instead of sequential row1d.
    // Trades some compression ratio for ~8x Lorenzo speedup.
    bool use_tile_uniform_eb = true;
};

extern "C" void run_gpu_huffman_u1_arrays(
    uint8_t* eq_U, uint8_t* eq_V, uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4]);

extern "C" void hf_u1_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint8_t* d_decoded, void* stream);

extern "C" void run_gpu_huffman_u2_u1_arrays(
    uint16_t* eq_U, uint16_t* eq_V,
    uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4]);

extern "C" void run_gpu_huffman_u2_arrays(
    uint16_t* eq_U, uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], float out_encode_ms[2], float out_decode_ms[2],
    uint8_t* out_h_blobs[2], float* out_parallel_wall_ms);

extern "C" void run_gpu_huffman_u2_arrays_timed(
    uint16_t* eq_U, uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], float out_encode_ms[2], float out_decode_ms[2],
    uint8_t* out_h_blobs[2], float* out_parallel_wall_ms, float* out_kernel_ms);

extern "C" void run_gpu_huffman_u2_u1_pipelined(
    uint16_t* eq_U, uint16_t* eq_V,
    uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* main_stream, void* bg_stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4]);

extern "C" void hf_u2_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint16_t* d_decoded, void* stream);

template<typename T, typename Eq2>
void verify_derive_eb_cpu_vs_gpu(const T* U, const T* V,
        const Eq2* eq_dEb_U, const Eq2* eq_dEb_V,
        size_t r1, size_t r2, T max_pwr_eb, T threshold) {
    printf("VERIFY_DERIVE_EB\n");
    size_t num_elements = r1 * r2;

    T * eb = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
    const T * U_pos = U;
    const T * V_pos = V;
    T * eb_pos = eb;
    const T X_upper[3][2] = {{0, 0}, {1, 0}, {1, 1}};
    const T X_lower[3][2] = {{0, 0}, {0, 1}, {1, 1}};
    const size_t offset_upper[3] = {0, r2, r2+1};
    const size_t offset_lower[3] = {0, 1, r2+1};
    printf("compute CPU eb\n");
    for(int i=0; i<r1-1; i++){
        const T * U_row_pos = U_pos;
        const T * V_row_pos = V_pos;
        T * eb_row_pos = eb_pos;
        for(int j=0; j<r2-1; j++){
            for(int k=0; k<2; k++){
                auto X = (k == 0) ? X_upper : X_lower;
                auto offset = (k == 0) ? offset_upper : offset_lower;
                (void) X;
                T max_cur_eb = max_eb_to_keep_position_and_type(U_row_pos[offset[0]], U_row_pos[offset[1]], U_row_pos[offset[2]],
                    V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]]);
                eb_row_pos[offset[0]] = MINF(eb_row_pos[offset[0]], max_cur_eb);
                eb_row_pos[offset[1]] = MINF(eb_row_pos[offset[1]], max_cur_eb);
                eb_row_pos[offset[2]] = MINF(eb_row_pos[offset[2]], max_cur_eb);
            }
            U_row_pos ++;
            V_row_pos ++;
            eb_row_pos ++;
        }
        U_pos += r2;
        V_pos += r2;
        eb_pos += r2;
    }
    printf("compute CPU eb done\n");

    const int base = 4;
    T log2_of_base = log2(base);
    T * eb_u = (T *) malloc(num_elements * sizeof(T));
    T * eb_v = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++){
        eb_u[i] = fabs(U[i]) * eb[i];
        int temp_eb_u = eb_exponential_quantize_2d(eb_u[i], base, log2_of_base, threshold);
        (void) temp_eb_u;
        if(eb_u[i] < threshold) eb_u[i] = 0;
    }
    for(int i=0; i<num_elements; i++){
        eb_v[i] = fabs(V[i]) * eb[i];
        int temp_eb_v = eb_exponential_quantize_2d(eb_v[i], base, log2_of_base, threshold);
        (void) temp_eb_v;
        if(eb_v[i] < threshold) eb_v[i] = 0;
    }

    Eq2 *h_eq_U = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    Eq2 *h_eq_V = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    cudaMemcpy(h_eq_U, eq_dEb_U, num_elements * sizeof(Eq2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eq_V, eq_dEb_V, num_elements * sizeof(Eq2), cudaMemcpyDeviceToHost);
    T * eb_u_gpu = (T *) malloc(num_elements * sizeof(T));
    T * eb_v_gpu = (T *) malloc(num_elements * sizeof(T));
    for (size_t i = 0; i < num_elements; i++) {
        eb_u_gpu[i] = h_eq_U[i] == 0 ? (T)0 : (T)(1ULL << (2 * h_eq_U[i])) * threshold;
        eb_v_gpu[i] = h_eq_V[i] == 0 ? (T)0 : (T)(1ULL << (2 * h_eq_V[i])) * threshold;
    }
    double diff = 0.0;
    double maxdiff = 0.0;
    int count=0;
    int maxdiff_index = 0;
    for (int i = 1; i < r1-1; i++){
        for(int j = 1; j < r2-1; j++){
            diff = fabs(eb_u_gpu[i*r2+j] - eb_u[i*r2+j]);
            if(diff > maxdiff)
            {
                maxdiff = diff;
                maxdiff_index = i*r2+j;
            }
            if (diff > std::numeric_limits<T>::epsilon()) {
                count++;
            }
        }
    }
    printf("maxdiff of u: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_u_gpu: %f, eb_u: %f\n", eb_u_gpu[maxdiff_index], eb_u[maxdiff_index]);

    diff = 0.0;
    maxdiff = 0.0;
    count=0;
    maxdiff_index = 0;
    for (int i = 1; i < r1-1; i++){
        for(int j = 1; j < r2-1; j++){
            diff = fabs(eb_v_gpu[i*r2+j] - eb_v[i*r2+j]);
            if(diff > maxdiff)
            {
                maxdiff = diff;
                maxdiff_index = i*r2+j;
            }
            if (diff > std::numeric_limits<T>::epsilon()) {
                count++;
            }
        }
    }
    printf("maxdiff of v: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_v_gpu: %f, eb_v: %f\n", eb_v_gpu[maxdiff_index], eb_v[maxdiff_index]);

    int count_zero = 0;
    for (int i = 0; i < r1 * r2; ++i) {
        if(eb_u_gpu[i] == 0) count_zero++;
    }
    printf("count_zero in the data: %d\n", count_zero);
    printf("\n");

    free(eb);
    free(eb_u);
    free(eb_v);
    free(eb_u_gpu);
    free(eb_v_gpu);
    free(h_eq_U);
    free(h_eq_V);
}

template<typename T, typename Eq2>
void verify_tile_error_bounds(
        const Eq2* eq_dEb_U, const Eq2* eq_dEb_V,
        const Eq2* tile_eq_dEb_U, const Eq2* tile_eq_dEb_V,
        size_t r1, size_t r2, T max_pwr_eb, T threshold) {
    printf("TEST_RIGHTNESS\n");
    size_t num_elements = r1 * r2;
    Eq2 *h_eq_dEb_U = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    Eq2 *h_eq_dEb_V = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    cudaMemcpy(h_eq_dEb_U, eq_dEb_U, r1 * r2 * sizeof(Eq2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eq_dEb_V, eq_dEb_V, r1 * r2 * sizeof(Eq2), cudaMemcpyDeviceToHost);
    size_t tile_cols = (r2 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM;
    size_t tile_rows = (r1 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM;
    size_t tile_count = tile_cols * tile_rows;
    Eq2 *h_tile_U = (Eq2 *) malloc(tile_count * sizeof(Eq2));
    Eq2 *h_tile_V = (Eq2 *) malloc(tile_count * sizeof(Eq2));
    cudaMemcpy(h_tile_U, tile_eq_dEb_U, tile_count * sizeof(Eq2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tile_V, tile_eq_dEb_V, tile_count * sizeof(Eq2), cudaMemcpyDeviceToHost);
    uint i = log2(max_pwr_eb / threshold)/2.0;
    printf("max_pwr_eb: %f, threshold: %f, i: %d\n", max_pwr_eb, threshold, i);
    size_t errors_U = 0, errors_V = 0;
    for (size_t ty = 0; ty < tile_rows; ty++) {
        for (size_t tx = 0; tx < tile_cols; tx++) {
            Eq2 min_U = std::numeric_limits<Eq2>::max();
            Eq2 min_V = std::numeric_limits<Eq2>::max();
            size_t y_end = std::min(r1, (ty + 1) * (size_t)LORENZO_TILE_DIM);
            size_t x_end = std::min(r2, (tx + 1) * (size_t)LORENZO_TILE_DIM);
            for (size_t y = ty * LORENZO_TILE_DIM; y < y_end; y++) {
                for (size_t x = tx * LORENZO_TILE_DIM; x < x_end; x++) {
                    size_t p = y * r2 + x;
                    min_U = std::min(min_U, h_eq_dEb_U[p]);
                    min_V = std::min(min_V, h_eq_dEb_V[p]);
                }
            }
            size_t tile_id = ty * tile_cols + tx;
            errors_U += h_tile_U[tile_id] != min_U;
            errors_V += h_tile_V[tile_id] != min_V;
        }
    }
    printf("tile EB U mismatches: %zu, V mismatches: %zu\n", errors_U, errors_V);
    free(h_eq_dEb_U);
    free(h_eq_dEb_V);
    free(h_tile_U);
    free(h_tile_V);
    printf("\n");
}

// compression with pre-computed error bounds
template<typename T, typename Eq1 = uint16_t, typename Eq2 = uint16_t>
unsigned char *
sz_compress_cp_preserve_2d_offline_gpu(const T * U, const T * V,
        Eq1 * eq_U, Eq1 * eq_V, Eq2 * eq_dEb,
        size_t r1, size_t r2, T max_pwr_eb,
        T* ot_val_U, uint32_t* ot_idx_U, uint32_t* ot_num_U, uint32_t* h_ot_num_U,
        T* ot_val_V, uint32_t* ot_idx_V, uint32_t* ot_num_V, uint32_t* h_ot_num_V,
        const char* out_fname = nullptr,
        const CpCompressDebugOptions& debug_options = CpCompressDebugOptions{},
        double* out_file_write_seconds = nullptr){
        

    size_t num_elements = r1 * r2;
    T *dU = nullptr, *dV = nullptr;
    T *row_dEb_U = nullptr, *row_dEb_V = nullptr;
    const T threshold = (T)(1.0 / (1 << 20));
    cudaStream_t stream;
    dim3 blockSize_v2(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 gridSize_v2((r2 + (blockSize_v2.x-2) - 1) / (blockSize_v2.x-2), 
        (r1 + (blockSize_v2.y-2)-1) / (blockSize_v2.y-2));
    dim3 blockSize_v3(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 gridSize_v3((r2 + (blockSize_v3.x-2) - 1) / (blockSize_v3.x-2), 
        (r1 + (blockSize_v3.y*NUM_PRE_THREAD-2)-1) / (blockSize_v3.y*NUM_PRE_THREAD-2));
    auto bytes = r1 * r2 * sizeof(T) * 2.0;
    auto GiB = 1024 * 1024 * 1024.0;
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + num_elements;
    T *ebIsZero_U_data = nullptr, *ebisZero_V_data = nullptr;
    uint32_t *ebIsZero_U_indices = nullptr, *ebIsZero_V_indices = nullptr;
    uint32_t *d_zeroeb_mask_U = nullptr, *d_zeroeb_mask_V = nullptr;
    int zero_eb_U_count, zero_eb_V_count;
    float ms_derive_eb = 0.0f, ms_land = 0.0f, ms_ebzero = 0.0f;
    float ms_uni = 0.0f, ms_row_eb = 0.0f, ms_lrz = 0.0f;
    cudaEvent_t _ce0 = nullptr, _ce1 = nullptr;
    cudaEventCreate(&_ce0); cudaEventCreate(&_ce1);
    size_t hf_lens[4] = {0, 0, 0, 0};
    float hf_encode_ms[4] = {0, 0, 0, 0};
    float hf_decode_ms[4] = {0, 0, 0, 0};
    float hf_wall_ms = 0.0f;
    float hf_kernel_ms = 0.0f;
    uint8_t* hf_blobs[4] = {nullptr, nullptr, nullptr, nullptr};

    Eq2* eq_dEb_V = nullptr;
    Eq2* tile_eq_dEb_U = nullptr;
    Eq2* tile_eq_dEb_V = nullptr;
    size_t tile_cols = (r2 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM;
    size_t tile_rows = (r1 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM;
    size_t tile_count = tile_cols * tile_rows;
    uint8_t* d_land_bitpack = nullptr;
    size_t land_bitpack_bytes = 0;
    size_t land_bitpack_capacity = (num_elements + 7) / 8;
    size_t zeroeb_mask_bytes = bitmask_word_bytes(num_elements);
    size_t land_count = 0;

    cudaMalloc(&dU, r1 * r2 * sizeof(T));
    cudaMalloc(&dV, r1 * r2 * sizeof(T));
    cudaMalloc(&eq_dEb_V, r1 * r2 * sizeof(Eq2));
    if (debug_options.use_tile_uniform_eb) {
        cudaMalloc(&tile_eq_dEb_U, tile_count * sizeof(Eq2));
        cudaMalloc(&tile_eq_dEb_V, tile_count * sizeof(Eq2));
    }
    cudaMalloc(&ebIsZero_U_data, r1 * r2 * sizeof(T));
    cudaMalloc(&ebIsZero_U_indices, r1 * r2 * sizeof(uint32_t));
    cudaMalloc(&ebisZero_V_data, r1 * r2 * sizeof(T));
    cudaMalloc(&ebIsZero_V_indices, r1 * r2 * sizeof(uint32_t));
    cudaMalloc(&d_zeroeb_mask_U, zeroeb_mask_bytes);
    cudaMalloc(&d_zeroeb_mask_V, zeroeb_mask_bytes);
    // Device counters for fused eb_zero kernel (reuse ot_num layout)
    uint32_t *d_zeb_U_cnt = nullptr, *d_zeb_V_cnt = nullptr;
    cudaMalloc(&d_zeb_U_cnt, sizeof(uint32_t));
    cudaMalloc(&d_zeb_V_cnt, sizeof(uint32_t));

    auto release_compression_workspace = [&]() {
        if (dU) { cudaFree(dU); dU = nullptr; }
        if (dV) { cudaFree(dV); dV = nullptr; }
        if (row_dEb_U) { cudaFree(row_dEb_U); row_dEb_U = nullptr; }
        if (row_dEb_V) { cudaFree(row_dEb_V); row_dEb_V = nullptr; }
        if (tile_eq_dEb_U) { cudaFree(tile_eq_dEb_U); tile_eq_dEb_U = nullptr; }
        if (tile_eq_dEb_V) { cudaFree(tile_eq_dEb_V); tile_eq_dEb_V = nullptr; }
        if (eq_dEb_V) { cudaFree(eq_dEb_V); eq_dEb_V = nullptr; }
        if (d_land_bitpack) { cudaFree(d_land_bitpack); d_land_bitpack = nullptr; }
        if (ebIsZero_U_data) { cudaFree(ebIsZero_U_data); ebIsZero_U_data = nullptr; }
        if (ebIsZero_U_indices) { cudaFree(ebIsZero_U_indices); ebIsZero_U_indices = nullptr; }
        if (ebisZero_V_data) { cudaFree(ebisZero_V_data); ebisZero_V_data = nullptr; }
        if (ebIsZero_V_indices) { cudaFree(ebIsZero_V_indices); ebIsZero_V_indices = nullptr; }
        if (d_zeroeb_mask_U) { cudaFree(d_zeroeb_mask_U); d_zeroeb_mask_U = nullptr; }
        if (d_zeroeb_mask_V) { cudaFree(d_zeroeb_mask_V); d_zeroeb_mask_V = nullptr; }
        if (d_zeb_U_cnt) { cudaFree(d_zeb_U_cnt); d_zeb_U_cnt = nullptr; }
        if (d_zeb_V_cnt) { cudaFree(d_zeb_V_cnt); d_zeb_V_cnt = nullptr; }
        if (_ce0) { cudaEventDestroy(_ce0); _ce0 = nullptr; }
        if (_ce1) { cudaEventDestroy(_ce1); _ce1 = nullptr; }
    };

    cudaMemcpy(dU, U, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    // Build land bitpack on CPU: bit[i]=1 where U[i]=V[i]=0
    {
        uint8_t* h_bp = new uint8_t[land_bitpack_capacity]();
        for (size_t i = 0; i < num_elements; i++) {
            if (U[i] == (T)0 && V[i] == (T)0) {
                h_bp[i/8] |= (uint8_t)(1u << (i%8));
                land_count++;
            }
        }
        if (land_count > 0) {
            land_bitpack_bytes = land_bitpack_capacity;
            cudaMalloc(&d_land_bitpack, land_bitpack_bytes);
            cudaMemcpy(d_land_bitpack, h_bp, land_bitpack_bytes, cudaMemcpyHostToDevice);
        }
        printf("land_count: %zu  land_bitpack_bytes: %zu\n", land_count, land_bitpack_bytes);
        delete[] h_bp;
    }
    

    //RUN_DERIVE_EB
    if(debug_options.run_derive_eb){
        printf("\nRUN_DERIVE_EB\n");
        printf("gridSize_v2: %d, %d\n", gridSize_v2.x, gridSize_v2.y);
        cudaEventRecord(_ce0);
        derive_eb_offline_v2<<<gridSize_v2, blockSize_v2>>>(
            dU, dV, eq_dEb, eq_dEb_V, r1, r2, max_pwr_eb);
        cudaEventRecord(_ce1);
        cudaEventSynchronize(_ce1);
        cudaEventElapsedTime(&ms_derive_eb, _ce0, _ce1);
        printf("\n");
        
        if(debug_options.verify_derive_eb){
            verify_derive_eb_cpu_vs_gpu(
                U, V, eq_dEb, eq_dEb_V, r1, r2, max_pwr_eb, threshold);
        }

        //DEAL_WITH_LAND
        if(debug_options.deal_with_land_data){
            //If U and V are both 0, means it is land in occean data, 
            //so it compress and decompress is all 0, we can just set eb = max-pwr
            printf("DEAL_WITH_LAND_DATA\n");
            // Precompute on CPU to avoid per-thread log2 on GPU
            const int land_id = (int)(log2(max_pwr_eb / threshold) / 2.0);
            cudaEventRecord(_ce0);
            thrust::for_each(
                idx_first, idx_last,
                [=] __device__ (size_t i) {
                    if (dU[i] == (T)0 && dV[i] == (T)0) {
                        eq_dEb[i]   = (Eq2)land_id;
                        eq_dEb_V[i] = (Eq2)land_id;
                    }
                }
            );
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_land, _ce0, _ce1);
            printf("finish deal with land data\n");
            printf("\n");
            
        }

        //deal with eb=zero (fused single-pass kernel: replaces 9 thrust calls)
        if(debug_options.deal_with_ebzero){
            printf("DEAL_WITH_EBZERO\n");
            int id = (int)(log2(max_pwr_eb / threshold) / 2.0);
            cudaMemset(d_zeb_U_cnt, 0, sizeof(uint32_t));
            cudaMemset(d_zeb_V_cnt, 0, sizeof(uint32_t));
            cudaMemset(d_zeroeb_mask_U, 0, zeroeb_mask_bytes);
            cudaMemset(d_zeroeb_mask_V, 0, zeroeb_mask_bytes);
            cudaEventRecord(_ce0);
            constexpr int ZEB_BLK = 512;
            int zeb_grid = ((int)num_elements + ZEB_BLK - 1) / ZEB_BLK;
            kernel_fused_ebzero<T, Eq2><<<zeb_grid, ZEB_BLK>>>(
                dU, dV, eq_dEb, eq_dEb_V,
                ebIsZero_U_indices, d_zeroeb_mask_U, d_zeb_U_cnt,
                ebIsZero_V_indices, d_zeroeb_mask_V, d_zeb_V_cnt,
                (Eq2)id, num_elements);
            thrust::exclusive_scan(thrust::device,
                ebIsZero_U_indices, ebIsZero_U_indices + num_elements,
                ebIsZero_U_indices);
            thrust::exclusive_scan(thrust::device,
                ebIsZero_V_indices, ebIsZero_V_indices + num_elements,
                ebIsZero_V_indices);
            kernel_compact_zeroeb_values<T><<<zeb_grid, ZEB_BLK>>>(
                dU, d_zeroeb_mask_U, ebIsZero_U_indices,
                ebIsZero_U_data, num_elements);
            kernel_compact_zeroeb_values<T><<<zeb_grid, ZEB_BLK>>>(
                dV, d_zeroeb_mask_V, ebIsZero_V_indices,
                ebisZero_V_data, num_elements);
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_ebzero, _ce0, _ce1);
            cudaMemcpy(&zero_eb_U_count, d_zeb_U_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&zero_eb_V_count, d_zeb_V_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("zero_eb_U_count: %d\n", zero_eb_U_count);
            printf("zero_eb_V_count: %d\n", zero_eb_V_count);
            printf("\n");
        }

        // Reduce to one EB exponent ID per tile. Lorenzo consumes this compact
        // array directly; no full-size float EB arrays are materialized.
        if(debug_options.use_tile_uniform_eb){
            printf("TILE_UNIFORM_EB (tile=%dx%d)\n", LORENZO_TILE_DIM, LORENZO_TILE_DIM);
            dim3 ublk(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1);
            dim3 ugrid((r2 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM,
                       (r1 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM, 1);
            cudaEventRecord(_ce0);
            kernel_reduce_tile_eb_pair<Eq2><<<ugrid, ublk>>>(
                eq_dEb, eq_dEb_V, tile_eq_dEb_U, tile_eq_dEb_V,
                (int)r1, (int)r2);
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_uni, _ce0, _ce1);
            printf("  uniformize eb: %.3f ms\n\n", ms_uni);
        }
        else {
            cudaMalloc(&row_dEb_U, num_elements * sizeof(T));
            cudaMalloc(&row_dEb_V, num_elements * sizeof(T));
            cudaEventRecord(_ce0);
            thrust::for_each(thrust::device, idx_first, idx_last, [=] __device__ (size_t i) {
                row_dEb_U[i] = (eq_dEb[i] == 0) ? (T)0
                    : (T)(1ULL << (2 * eq_dEb[i])) * threshold;
                row_dEb_V[i] = (eq_dEb_V[i] == 0) ? (T)0
                    : (T)(1ULL << (2 * eq_dEb_V[i])) * threshold;
            });
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_row_eb, _ce0, _ce1);
        }

        if(debug_options.run_quantization_lorenzo){
            printf("RUN_QUANTIZATION_LORENZO\n");
            cudaMemset(ot_num_U, 0, sizeof(uint32_t));
            cudaMemset(ot_num_V, 0, sizeof(uint32_t));
            *h_ot_num_U = 0, *h_ot_num_V=0;
            cudaMemset(eq_U, 0, r2 * r1 * sizeof(Eq1));
            cudaMemset(eq_V, 0, r2 * r1 * sizeof(Eq1));

            float lrz_U_ms = 0.0f, lrz_V_ms = 0.0f;
            if (debug_options.use_tile_uniform_eb) {
                // Parallel prefix-sum tile Lorenzo (uniform eb per tile).
                // Outliers stored as delta-codes (candidate=delta+radius), pre-scatter convention.
                psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__tile_eb<T, Eq1, Eq2>(
                    dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U,
                    tile_eq_dEb_U, threshold, RADIUS, &lrz_U_ms, 0);
                psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__tile_eb<T, Eq1, Eq2>(
                    dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V,
                    tile_eq_dEb_V, threshold, RADIUS, &lrz_V_ms, 0);
            } else {
                // Sequential row1d Lorenzo (exact-value outliers, post-scatter convention).
                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                    dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U,
                    row_dEb_U, RADIUS, &lrz_U_ms, 0);
                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                    dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V,
                    row_dEb_V, RADIUS, &lrz_V_ms, 0);
            }
            ms_lrz = lrz_U_ms + lrz_V_ms;
            cudaDeviceSynchronize();  // ensure ot_num arrays are ready for memcpy
            cudaMemcpy(h_ot_num_U, ot_num_U, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ot_num_V, ot_num_V, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("ot_num_U: %d, ot_num_V: %d\n", *h_ot_num_U, *h_ot_num_V);
            if(debug_options.run_hf){
                size_t raw_eq  = num_elements * sizeof(Eq1);
                size_t raw_deb = num_elements * sizeof(Eq2);
                printf("RUN_HF\n");
                printf("  before HF: eq_U=%zu  eq_V=%zu  eq_dEb=%zu bytes\n",
                       raw_eq, raw_eq, raw_deb);
                cudaStreamCreate(&stream);
                if (debug_options.use_tile_uniform_eb && sizeof(Eq1) == sizeof(uint16_t)) {
                    size_t eq_hf_lens[2] = {0, 0};
                    float eq_hf_encode_ms[2] = {0.0f, 0.0f};
                    float eq_hf_decode_ms[2] = {0.0f, 0.0f};
                    uint8_t* eq_hf_blobs[2] = {nullptr, nullptr};
                    run_gpu_huffman_u2_arrays_timed(
                        reinterpret_cast<uint16_t*>(eq_U), reinterpret_cast<uint16_t*>(eq_V),
                        num_elements, stream,
                        eq_hf_lens, eq_hf_encode_ms, eq_hf_decode_ms,
                        eq_hf_blobs, &hf_wall_ms, &hf_kernel_ms);
                    hf_lens[0] = eq_hf_lens[0];
                    hf_lens[1] = eq_hf_lens[1];
                    hf_encode_ms[0] = eq_hf_encode_ms[0];
                    hf_encode_ms[1] = eq_hf_encode_ms[1];
                    hf_decode_ms[0] = eq_hf_decode_ms[0];
                    hf_decode_ms[1] = eq_hf_decode_ms[1];
                    hf_blobs[0] = eq_hf_blobs[0];
                    hf_blobs[1] = eq_hf_blobs[1];

                    copy_tile_eb_payloads<Eq2>(
                        tile_eq_dEb_U, tile_eq_dEb_V, tile_count,
                        hf_blobs[2], hf_blobs[3],
                        hf_lens[2], hf_lens[3]);

                    printf("  after  HF/pack: eq_U=%zu (%.1f%%)  eq_V=%zu (%.1f%%)  tile_dEb_U=%zu  tile_dEb_V=%zu\n",
                           hf_lens[0], hf_lens[0]*100.0/raw_eq,
                           hf_lens[1], hf_lens[1]*100.0/raw_eq,
                           hf_lens[2], hf_lens[3]);
                    printf("  tile EB payload: %zu tiles/field\n",
                           hf_lens[2] / sizeof(Eq2));
                }
                else if (sizeof(Eq1) == sizeof(uint16_t) && sizeof(Eq2) == sizeof(uint8_t)) {
                    run_gpu_huffman_u2_u1_arrays(
                        reinterpret_cast<uint16_t*>(eq_U), reinterpret_cast<uint16_t*>(eq_V),
                        reinterpret_cast<uint8_t*>(eq_dEb), reinterpret_cast<uint8_t*>(eq_dEb_V),
                        num_elements, stream,
                        hf_lens, hf_encode_ms, hf_decode_ms,
                        hf_blobs);
                    hf_wall_ms = hf_encode_ms[0] + hf_encode_ms[1] +
                                 hf_encode_ms[2] + hf_encode_ms[3];
                    hf_kernel_ms = hf_wall_ms;
                    printf("  after  HF: eq_U=%zu (%.1f%%)  eq_V=%zu (%.1f%%)  eq_dEb_U=%zu (%.1f%%)  eq_dEb_V=%zu (%.1f%%)\n",
                           hf_lens[0], hf_lens[0]*100.0/raw_eq,
                           hf_lens[1], hf_lens[1]*100.0/raw_eq,
                           hf_lens[2], hf_lens[2]*100.0/raw_deb,
                           hf_lens[3], hf_lens[3]*100.0/raw_deb);
                }
                else if (sizeof(Eq1) == sizeof(uint8_t) && sizeof(Eq2) == sizeof(uint8_t)) {
                    run_gpu_huffman_u1_arrays(
                        reinterpret_cast<uint8_t*>(eq_U), reinterpret_cast<uint8_t*>(eq_V),
                        reinterpret_cast<uint8_t*>(eq_dEb), reinterpret_cast<uint8_t*>(eq_dEb_V),
                        num_elements, stream,
                        hf_lens, hf_encode_ms, hf_decode_ms,
                        hf_blobs);
                    hf_wall_ms = hf_encode_ms[0] + hf_encode_ms[1] +
                                 hf_encode_ms[2] + hf_encode_ms[3];
                    hf_kernel_ms = hf_wall_ms;
                    printf("  after  HF: eq_U=%zu (%.1f%%)  eq_V=%zu (%.1f%%)  eq_dEb_U=%zu (%.1f%%)  eq_dEb_V=%zu (%.1f%%)\n",
                           hf_lens[0], hf_lens[0]*100.0/raw_eq,
                           hf_lens[1], hf_lens[1]*100.0/raw_eq,
                           hf_lens[2], hf_lens[2]*100.0/raw_deb,
                           hf_lens[3], hf_lens[3]*100.0/raw_deb);
                }
                else {
                    printf("RUN_HF: unsupported Eq type combination.\n\n");
                }
                cudaStreamDestroy(stream);

                {
                    float total_ms = ms_derive_eb + ms_land + ms_ebzero + ms_uni +
                                     ms_row_eb + ms_lrz + hf_kernel_ms;
                    printf("COMPRESS_TIME (GPU kernels only)\n");
                    printf("  derive_eb : %8.3f ms\n", ms_derive_eb);
                    printf("  land_data : %8.3f ms\n", ms_land);
                    printf("  eb_zero   : %8.3f ms\n", ms_ebzero);
                    printf("  uniform_eb: %8.3f ms\n", ms_uni);
                    printf("  row_eb    : %8.3f ms\n", ms_row_eb);
                    printf("  lorenzo   : %8.3f ms\n", ms_lrz);
                    printf("  huffman   : %8.3f ms\n", hf_kernel_ms);
                    printf("  total     : %8.3f ms  (%.2f GiB/s)\n\n",
                           total_ms, bytes / GiB / (total_ms / 1000));
                }

                // Write compressed data: copy ot and zeroeb arrays from GPU to host, then write file
                if (out_fname && hf_blobs[0] && hf_blobs[1] && hf_blobs[2] && hf_blobs[3]) {
                    cudaMemcpy(h_ot_num_U, ot_num_U, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_ot_num_V, ot_num_V, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    uint32_t cnt_ot_U = *h_ot_num_U, cnt_ot_V = *h_ot_num_V;
                    uint32_t cnt_zeb_U = (uint32_t)zero_eb_U_count, cnt_zeb_V = (uint32_t)zero_eb_V_count;

                    printf("  special_count_U=%u  (outlier=%u + zeroeb=%u)\n",
                           cnt_ot_U + cnt_zeb_U, cnt_ot_U, cnt_zeb_U);
                    printf("  special_count_V=%u  (outlier=%u + zeroeb=%u)\n",
                           cnt_ot_V + cnt_zeb_V, cnt_ot_V, cnt_zeb_V);

	                    uint32_t* h_ot_idx_U  = new uint32_t[cnt_ot_U  ? cnt_ot_U  : 1];
	                    float*    h_ot_val_U  = new float   [cnt_ot_U  ? cnt_ot_U  : 1];
	                    uint8_t*  h_zeb_mask_U = new uint8_t[cnt_zeb_U ? zeroeb_mask_bytes : 1];
	                    float*    h_zeb_val_U = new float   [cnt_zeb_U ? cnt_zeb_U : 1];
	                    uint32_t* h_ot_idx_V  = new uint32_t[cnt_ot_V  ? cnt_ot_V  : 1];
	                    float*    h_ot_val_V  = new float   [cnt_ot_V  ? cnt_ot_V  : 1];
	                    uint8_t*  h_zeb_mask_V = new uint8_t[cnt_zeb_V ? zeroeb_mask_bytes : 1];
	                    float*    h_zeb_val_V = new float   [cnt_zeb_V ? cnt_zeb_V : 1];

                    if (cnt_ot_U) {
                        cudaMemcpy(h_ot_idx_U, ot_idx_U, cnt_ot_U * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_ot_val_U, ot_val_U, cnt_ot_U * sizeof(float),    cudaMemcpyDeviceToHost);
	                    }
	                    if (cnt_zeb_U) {
	                        cudaMemcpy(h_zeb_mask_U, d_zeroeb_mask_U, zeroeb_mask_bytes, cudaMemcpyDeviceToHost);
	                        cudaMemcpy(h_zeb_val_U, ebIsZero_U_data,    cnt_zeb_U * sizeof(float),    cudaMemcpyDeviceToHost);
	                    }
                    if (cnt_ot_V) {
                        cudaMemcpy(h_ot_idx_V, ot_idx_V, cnt_ot_V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_ot_val_V, ot_val_V, cnt_ot_V * sizeof(float),    cudaMemcpyDeviceToHost);
	                    }
	                    if (cnt_zeb_V) {
	                        cudaMemcpy(h_zeb_mask_V, d_zeroeb_mask_V, zeroeb_mask_bytes, cudaMemcpyDeviceToHost);
	                        cudaMemcpy(h_zeb_val_V, ebisZero_V_data,    cnt_zeb_V * sizeof(float),    cudaMemcpyDeviceToHost);
	                    }

                    uint8_t* h_land_bitpack = nullptr;
                    if (land_bitpack_bytes > 0) {
                        h_land_bitpack = new uint8_t[land_bitpack_bytes];
                        cudaMemcpy(h_land_bitpack, d_land_bitpack, land_bitpack_bytes, cudaMemcpyDeviceToHost);
                    }

                    double file_write_seconds = write_cucpsz(out_fname, r1, r2, max_pwr_eb,
                        hf_blobs, hf_lens,
	                        cnt_ot_U,  h_ot_idx_U,  h_ot_val_U,
	                        cnt_zeb_U, h_zeb_mask_U, h_zeb_val_U,
	                        cnt_ot_V,  h_ot_idx_V,  h_ot_val_V,
	                        cnt_zeb_V, h_zeb_mask_V, h_zeb_val_V,
                        h_land_bitpack, land_bitpack_bytes,
                        debug_options.use_tile_uniform_eb ? 1u : 0u,
                        debug_options.use_tile_uniform_eb ? (uint32_t)LORENZO_TILE_DIM : 0u);
                    if (out_file_write_seconds) {
                        *out_file_write_seconds = file_write_seconds;
                    }
                    delete[] h_land_bitpack;

	                    delete[] h_ot_idx_U; delete[] h_ot_val_U;
	                    delete[] h_zeb_mask_U; delete[] h_zeb_val_U;
	                    delete[] h_ot_idx_V; delete[] h_ot_val_V;
	                    delete[] h_zeb_mask_V; delete[] h_zeb_val_V;
                    for (int i = 0; i < 4; i++) { delete[] hf_blobs[i]; hf_blobs[i] = nullptr; }
                }
            }

            if(debug_options.compute_ratio){
                printf("COMPUTE_RATIO\n");
                size_t zeroeb_mask_bytes_U = zero_eb_U_count ? zeroeb_mask_bytes : 0;
                size_t zeroeb_mask_bytes_V = zero_eb_V_count ? zeroeb_mask_bytes : 0;
                size_t sp_U_bytes = (*h_ot_num_U) * (sizeof(uint32_t) + sizeof(float)) +
                                    zeroeb_mask_bytes_U + (uint32_t)zero_eb_U_count * sizeof(float);
                size_t sp_V_bytes = (*h_ot_num_V) * (sizeof(uint32_t) + sizeof(float)) +
                                    zeroeb_mask_bytes_V + (uint32_t)zero_eb_V_count * sizeof(float);
                size_t original = r1 * r2 * sizeof(T);
                size_t compressed_U = hf_lens[0] + hf_lens[2] + sp_U_bytes;
                size_t compressed_V = hf_lens[1] + hf_lens[3] + sp_V_bytes;
                size_t compressed_payload = compressed_U + compressed_V;
                size_t compressed_file = sizeof(CucpszHeader) + land_bitpack_bytes + compressed_payload;
                printf("  outlier_U=%u  zeroeb_U=%d  outlier_V=%u  zeroeb_V=%d\n",
                       *h_ot_num_U, zero_eb_U_count, *h_ot_num_V, zero_eb_V_count);
                printf("U compression ratio: %f\n", (double)original / compressed_U);
                printf("V compression ratio: %f\n", (double)original / compressed_V);
                printf("Overall compression ratio: %f\n", (double)(original * 2) / compressed_file);
                printf("  compressed payload=%zu  file_estimate=%zu  land=%zu  header=%zu\n",
                       compressed_payload, compressed_file, land_bitpack_bytes, sizeof(CucpszHeader));
                printf("\n");
            }

            if(debug_options.write_debug_data){
                printf("WRITE_debug_DATA\n");
                //write the eq_U eq_V
                // 1. 从 GPU 拷贝到 Host
                Eq1* h_eq_U = new Eq1[r2 * r1];
                Eq1* h_eq_V = new Eq1[r2 * r1];
                Eq2* h_eq_dEb = new Eq2[r2 * r1];
                cudaMemcpy(h_eq_U, eq_U, r2 * r1 * sizeof(Eq1), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_eq_V, eq_V, r2 * r1 * sizeof(Eq1), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_eq_dEb, eq_dEb, r2 * r1 * sizeof(Eq2), cudaMemcpyDeviceToHost);
                if(debug_options.max_data){
                    printf("MAX_DATA\n");
                    int max_eq_U = 0, max_eq_V = 0;
                    for(int i = 0; i < num_elements; i++){
                        if(h_eq_U[i] > max_eq_U){
                            max_eq_U = h_eq_U[i];
                        }
                        if(h_eq_V[i] > max_eq_V){
                            max_eq_V = h_eq_V[i];
                        }
                    }
                    int max_eq_dEb = 0;
                    for(int i = 0; i < num_elements; i++){
                        if(h_eq_dEb[i] > max_eq_dEb){
                            max_eq_dEb = h_eq_dEb[i];
                        }
                    }
                    printf("max_eq_U: %d, max_eq_V: %d\n", max_eq_U, max_eq_V);
                    printf("max_eq_dEb: %d\n", max_eq_dEb);
                }

                // 2. 写入 .dat 文件
                std::ofstream outU("eq_U.dat", std::ios::out | std::ios::binary);
                outU.write(reinterpret_cast<char*>(h_eq_U), r2 * r1 * sizeof(Eq1));
                outU.close();
                std::ofstream outEB("eq_dEb.dat", std::ios::out | std::ios::binary);
                outEB.write(reinterpret_cast<char*>(h_eq_dEb), r2 * r1 * sizeof(Eq2));
                outEB.close();
                std::ofstream outV("eq_V.dat", std::ios::out | std::ios::binary);
                outV.write(reinterpret_cast<char*>(h_eq_V), r2 * r1 * sizeof(Eq1));
                outV.close();
                // 3. 释放内存
                delete[] h_eq_U;
                delete[] h_eq_V;
                delete[] h_eq_dEb;
                printf("finished write data\n");
                printf("\n");
            }

            printf("\n");
        }
    }  

    for (int i = 0; i < 4; i++) {
        delete[] hf_blobs[i];
        hf_blobs[i] = nullptr;
    }
    release_compression_workspace();
    return nullptr;

}


int main(int argc, char ** argv){
    if (argc != 6) {
        fprintf(stderr, "Usage: %s U.bin V.bin r1 r2 max_relative_eb\n", argv[0]);
        return 1;
    }

    const size_t r1 = strtoull(argv[3], nullptr, 10);
    const size_t r2 = strtoull(argv[4], nullptr, 10);
    const float max_eb = strtof(argv[5], nullptr);
    if (r1 == 0 || r2 == 0 || r1 > SIZE_MAX / r2) {
        fprintf(stderr, "Invalid dimensions: %s x %s\n", argv[3], argv[4]);
        return 1;
    }

    const size_t expected_elements = r1 * r2;
    using IoClock = std::chrono::steady_clock;
    size_t u_elements = 0, v_elements = 0;
    auto read_start = IoClock::now();
    float * U = readfile<float>(argv[1], u_elements);
    auto read_u_done = IoClock::now();
    float * V = readfile<float>(argv[2], v_elements);
    auto read_v_done = IoClock::now();
    const double read_u_seconds =
        std::chrono::duration<double>(read_u_done - read_start).count();
    const double read_v_seconds =
        std::chrono::duration<double>(read_v_done - read_u_done).count();
    if (!U || !V || u_elements != expected_elements || v_elements != expected_elements) {
        fprintf(stderr,
            "Input size mismatch: expected %zu floats, U has %zu, V has %zu\n",
            expected_elements, u_elements, v_elements);
        free(U);
        free(V);
        return 1;
    }

    CpCompressDebugOptions debug_options;

    cout << "start Compression\n";

    OtData ot_U;
    allocOtData(ot_U, expected_elements);
    OtData ot_V;
    allocOtData(ot_V, expected_elements);
    uint16_t *eq_U, *eq_V;
    cudaMalloc(&eq_U, expected_elements * sizeof(uint16_t));
    cudaMalloc(&eq_V, expected_elements * sizeof(uint16_t));
    uint8_t *eq_dEb;
    cudaMalloc(&eq_dEb, expected_elements * sizeof(uint8_t));

    string cucpsz_fname = string(argv[1]) + ".cucpsz";
    double file_write_seconds = 0.0;

    unsigned char * result = sz_compress_cp_preserve_2d_offline_gpu(U, V,  eq_U, eq_V, eq_dEb, r1,  r2, max_eb,
        ot_U.val, ot_U.idx, ot_U.num, ot_U.h_num, ot_V.val, ot_V.idx, ot_V.num, ot_V.h_num,
        cucpsz_fname.c_str(), debug_options, &file_write_seconds);

    free(result);
    free(U);
    free(V);
    freeOtData(ot_U);
    freeOtData(ot_V);
    cudaFree(eq_U);
    cudaFree(eq_V);
    cudaFree(eq_dEb);
    printf("FILE_IO_TIME\n");
    printf("  read_U : %.6f s\n", read_u_seconds);
    printf("  read_V : %.6f s\n", read_v_seconds);
    printf("  read   : %.6f s\n", read_u_seconds + read_v_seconds);
    if (file_write_seconds >= 0.0) {
        printf("  write  : %.6f s\n", file_write_seconds);
        printf("  total  : %.6f s\n", read_u_seconds + read_v_seconds + file_write_seconds);
    }
    else {
        printf("  write  : failed\n");
    }
    return 0;
}
