#include "sz_cp_preserve_utils.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz3_utils.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"
#include <chrono>
#include <limits>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/numeric/eigen_solver2.hh>
#include <ftk/numeric/critical_point_type.hh>
#include <complex>
#include <unordered_map>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include "kernel/lrz/lproto.hh"
#include "lorenzo_tile_dim.h"
#include "fused_ebzero_kernel.cuh"
#include "tile_uniform_eb.cuh"

using namespace std;

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 16
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
__global__ void derive_eb_offline_v2(const T* __restrict__ dU, const T* __restrict__ dV, T* __restrict__ dEb, T* __restrict__  dEb_U,  T* __restrict__ dEb_V, 
        Eq2* __restrict eq_dEb_U, Eq2* __restrict eq_dEb_V, int r1, int r2, T max_pwr_eb){
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

        dEb_U[(row+1) * r2 + (col+1)] = temp;
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

        dEb_V[(row+1) * r2 + (col+1)] =  temp;
        eq_dEb_V[(row+1) * r2 + (col+1)] = id;
    }
    __syncthreads();

    if((row == 0 || col ==0 || row==r1-1 || col == r2-1)&&(row<r1-1 && col<r2-1)){
        dEb_U[row * r2 + col] = 0;
        dEb_V[row * r2 + col] = 0;
        eq_dEb_U[row * r2 + col] = 0;
        eq_dEb_V[row * r2 + col] = 0;
    }
    __syncthreads();
}

template <typename Eq2>
__global__ void kernel_pack_tile_eq_eb(
    const Eq2* __restrict__ eq_dEb,
    Eq2* __restrict__ tile_eq_dEb,
    int r1, int r2, int tile_dim, int tile_cols)
{
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_rows = (r1 + tile_dim - 1) / tile_dim;
    if (tile_x >= tile_cols || tile_y >= tile_rows) return;

    int x = tile_x * tile_dim;
    int y = tile_y * tile_dim;
    tile_eq_dEb[tile_y * tile_cols + tile_x] = eq_dEb[(size_t)y * r2 + x];
}

template <typename T, typename Eq2>
__global__ void kernel_expand_tile_eq_eb(
    const Eq2* __restrict__ tile_eq_dEb,
    T* __restrict__ dEb,
    int r1, int r2, int tile_dim, int tile_cols, T threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= r2 || y >= r1) return;

    Eq2 eb_id = tile_eq_dEb[(y / tile_dim) * tile_cols + (x / tile_dim)];
    dEb[(size_t)y * r2 + x] = (eb_id == 0)
        ? (T)0
        : (T)(1ULL << (2 * (unsigned)eb_id)) * threshold;
}

template <typename Eq2>
static void pack_tile_eb_payloads(
    const Eq2* d_eq_dEb_U, const Eq2* d_eq_dEb_V,
    size_t r1, size_t r2,
    uint8_t*& h_tile_U, uint8_t*& h_tile_V,
    size_t& tile_bytes_U, size_t& tile_bytes_V,
    float& ms_pack)
{
    int tile_dim = LORENZO_TILE_DIM;
    int tile_cols = ((int)r2 + tile_dim - 1) / tile_dim;
    int tile_rows = ((int)r1 + tile_dim - 1) / tile_dim;
    size_t tile_count = (size_t)tile_cols * tile_rows;
    tile_bytes_U = tile_count * sizeof(Eq2);
    tile_bytes_V = tile_count * sizeof(Eq2);

    Eq2 *d_tile_U = nullptr, *d_tile_V = nullptr;
    cudaMalloc(&d_tile_U, tile_bytes_U);
    cudaMalloc(&d_tile_V, tile_bytes_V);

    dim3 block(16, 16, 1);
    dim3 grid((tile_cols + block.x - 1) / block.x,
              (tile_rows + block.y - 1) / block.y, 1);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    kernel_pack_tile_eq_eb<Eq2><<<grid, block>>>(d_eq_dEb_U, d_tile_U, (int)r1, (int)r2, tile_dim, tile_cols);
    kernel_pack_tile_eq_eb<Eq2><<<grid, block>>>(d_eq_dEb_V, d_tile_V, (int)r1, (int)r2, tile_dim, tile_cols);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms_pack, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    h_tile_U = new uint8_t[tile_bytes_U ? tile_bytes_U : 1];
    h_tile_V = new uint8_t[tile_bytes_V ? tile_bytes_V : 1];
    cudaMemcpy(h_tile_U, d_tile_U, tile_bytes_U, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tile_V, d_tile_V, tile_bytes_V, cudaMemcpyDeviceToHost);

    cudaFree(d_tile_U);
    cudaFree(d_tile_V);
}

// Header is naturally padded to 96 bytes on the current ABI.
// File layout after header:
//   [payload 0: HF eq_U]
//   [payload 1: HF eq_V]
//   [payload 2: HF eq_dEb_U for row1d, raw per-tile eq_dEb_U for tile variant]
//   [payload 3: HF eq_dEb_V for row1d, raw per-tile eq_dEb_V for tile variant]
//   [ot_idx_U: ot_count_U * 4 bytes]
//   [ot_val_U: ot_count_U * 4 bytes]
//   [zeroeb_idx_U: zeroeb_count_U * 4 bytes]
//   [zeroeb_val_U: zeroeb_count_U * 4 bytes]
//   [same 4 arrays for V]
struct CucpszHeader {
    char magic[8];       // "CUCPSZ\0\0"
    uint64_t r1, r2;
    float max_pwr_eb;
    uint32_t ot_count_U, ot_count_V;        // outlier counts
    uint32_t zeroeb_count_U, zeroeb_count_V; // zero-EB counts
    uint64_t hf_blob_len[4];  // eq_U, eq_V, eq_dEb_U/tile_U, eq_dEb_V/tile_V
    uint64_t land_bitpack_bytes; // bytes in land bitpack (U=V=0 positions)
    uint32_t lorenzo_variant;    // 0 = row1d (exact-val outliers), 1 = tile prefix-sum (delta-code outliers)
    uint32_t lorenzo_tile_dim;   // valid when lorenzo_variant == 1
};

// Write the .cucpsz file.
// Per field: ot_idx+ot_val (outlier quant codes for pre-scatter),
//            zeroeb_idx+zeroeb_val (original floats for post-scatter).
static void write_cucpsz(
    const char* fname,
    size_t r1, size_t r2, float max_pwr_eb,
    uint8_t* hf_blobs[4], size_t hf_lens[4],
    uint32_t ot_count_U,    const uint32_t* ot_idx_U,    const float* ot_val_U,
    uint32_t zeroeb_count_U,const uint32_t* zeroeb_idx_U,const float* zeroeb_val_U,
    uint32_t ot_count_V,    const uint32_t* ot_idx_V,    const float* ot_val_V,
    uint32_t zeroeb_count_V,const uint32_t* zeroeb_idx_V,const float* zeroeb_val_V,
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

    FILE* f = fopen(fname, "wb");
    fwrite(&hdr, sizeof(hdr), 1, f);

    size_t total_bytes = sizeof(hdr) + land_bitpack_bytes;
    for (int i = 0; i < 4; i++) total_bytes += hf_lens[i];
    total_bytes += (ot_count_U + zeroeb_count_U) * (sizeof(uint32_t) + sizeof(float));
    total_bytes += (ot_count_V + zeroeb_count_V) * (sizeof(uint32_t) + sizeof(float));
    auto pct = [&](size_t bytes) { return bytes * 100.0 / total_bytes; };

    const char* hf_names[4] = {
        "eq_U",
        "eq_V",
        lorenzo_variant == 1u ? "tile_dEb_U" : "eq_dEb_U",
        lorenzo_variant == 1u ? "tile_dEb_V" : "eq_dEb_V"
    };
    for (int i = 0; i < 4; i++) {
        fwrite(hf_blobs[i], 1, hf_lens[i], f);
        printf("  payload %-10s : %zu bytes (%.1f%%)\n", hf_names[i], hf_lens[i], pct(hf_lens[i]));
    }

    // Land bitpack (1 bit per element, 1 = U=V=0)
    if (land_bitpack_bytes > 0) fwrite(land_bitpack, 1, land_bitpack_bytes, f);
    printf("  land bitpack       : %zu bytes (%.1f%%)\n", land_bitpack_bytes, pct(land_bitpack_bytes));

    // U: outlier indices+values, then zeroeb indices+original_floats
    fwrite(ot_idx_U,    sizeof(uint32_t), ot_count_U,    f);
    fwrite(ot_val_U,    sizeof(float),    ot_count_U,    f);
    fwrite(zeroeb_idx_U,sizeof(uint32_t), zeroeb_count_U,f);
    fwrite(zeroeb_val_U,sizeof(float),    zeroeb_count_U,f);
    size_t sp_U_bytes = (ot_count_U + zeroeb_count_U) * (sizeof(uint32_t) + sizeof(float));
    printf("  special U (ot=%u + zeroeb=%u): %zu bytes (%.1f%%)\n",
           ot_count_U, zeroeb_count_U, sp_U_bytes, pct(sp_U_bytes));

    // V
    fwrite(ot_idx_V,    sizeof(uint32_t), ot_count_V,    f);
    fwrite(ot_val_V,    sizeof(float),    ot_count_V,    f);
    fwrite(zeroeb_idx_V,sizeof(uint32_t), zeroeb_count_V,f);
    fwrite(zeroeb_val_V,sizeof(float),    zeroeb_count_V,f);
    size_t sp_V_bytes = (ot_count_V + zeroeb_count_V) * (sizeof(uint32_t) + sizeof(float));
    printf("  special V (ot=%u + zeroeb=%u): %zu bytes (%.1f%%)\n",
           ot_count_V, zeroeb_count_V, sp_V_bytes, pct(sp_V_bytes));

    printf("  total              : %zu bytes\n", total_bytes);
    fclose(f);
    printf("Written %s\n", fname);
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
    bool write_data = true;
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
    uint8_t* out_h_blobs[2]);

extern "C" void run_gpu_huffman_u2_u1_pipelined(
    uint16_t* eq_U, uint16_t* eq_V,
    uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* main_stream, void* bg_stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4]);

extern "C" void hf_u2_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint16_t* d_decoded, void* stream);

template<typename T>
void verify_derive_eb_cpu_vs_gpu(const T* U, const T* V, const T* dEb_U, const T* dEb_V,
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

    T * eb_u_gpu = (T *) malloc(num_elements * sizeof(T));
    cudaMemcpy(eb_u_gpu, dEb_U, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    T * eb_v_gpu = (T *) malloc(num_elements * sizeof(T));
    cudaMemcpy(eb_v_gpu, dEb_V, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
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
}

template<typename T, typename Eq2>
void verify_decompressed_error_bounds(const T* dEb_U, const T* dEb_V, const Eq2* eq_dEb_U, const Eq2* eq_dEb_V,
        size_t r1, size_t r2, T max_pwr_eb, T threshold) {
    printf("TEST_RIGHTNESS\n");
    size_t num_elements = r1 * r2;
    T * h_EB_U_decomp = (T *) malloc(num_elements * sizeof(T));
    T * h_EB_V_decomp = (T *) malloc(num_elements * sizeof(T));
    Eq2 *h_eq_dEb_U = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    Eq2 *h_eq_dEb_V = (Eq2 *) malloc(num_elements * sizeof(Eq2));
    cudaMemcpy(h_eq_dEb_U, eq_dEb_U, r1 * r2 * sizeof(Eq2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eq_dEb_V, eq_dEb_V, r1 * r2 * sizeof(Eq2), cudaMemcpyDeviceToHost);
    uint i = log2(max_pwr_eb / threshold)/2.0;
    printf("max_pwr_eb: %f, threshold: %f, i: %d\n", max_pwr_eb, threshold, i);
    for(int i = 0; i < r1 * r2; i++){
        if(h_eq_dEb_U[i] == 0){
            h_EB_U_decomp[i] = 0;
        }
        if(h_eq_dEb_U[i] != 0){
            h_EB_U_decomp[i] = (T)(1ULL << (2 * h_eq_dEb_U[i])) * threshold;
        }
        if(h_eq_dEb_V[i] == 0){
            h_EB_V_decomp[i] = 0;
        }
        if(h_eq_dEb_V[i] != 0){
            h_EB_V_decomp[i] = (T)(1ULL << (2 * h_eq_dEb_V[i])) * threshold;
        }
    }
    T * eb_u_gpu = (T *) malloc(num_elements * sizeof(T));
    cudaMemcpy(eb_u_gpu, dEb_U, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    T * eb_v_gpu = (T *) malloc(num_elements * sizeof(T));
    cudaMemcpy(eb_v_gpu, dEb_V, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);

    double diff = 0.0;
    double maxdiff = 0.0;
    int count=0;
    int maxdiff_index = 0;
    for (int i = 0; i < r1; i++){
        for(int j = 0; j < r2; j++){
            diff = fabs(eb_u_gpu[i*r2+j] - h_EB_U_decomp[i*r2+j]);
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
    printf("eb_u_gpu: %f, EB_U_decomp: %f\n", eb_u_gpu[maxdiff_index], h_EB_U_decomp[maxdiff_index]);
    diff = 0.0;
    maxdiff = 0.0;
    count=0;
    maxdiff_index = 0;
    for (int i = 0; i < r1; i++){
        for(int j = 0; j < r2; j++){
            diff = fabs(eb_v_gpu[i*r2+j] - h_EB_V_decomp[i*r2+j]);
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
    printf("eb_v_gpu: %f, EB_V_decomp: %f\n", eb_v_gpu[maxdiff_index], h_EB_V_decomp[maxdiff_index]);
    free(h_EB_U_decomp);
    free(h_EB_V_decomp);
    free(h_eq_dEb_U);
    free(h_eq_dEb_V);
    free(eb_u_gpu);
    free(eb_v_gpu);
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
        T* U_decomp, T* V_decomp,
        const char* out_fname = nullptr){
    const CpCompressDebugOptions debug_options;
        

    size_t num_elements = r1 * r2;
    T *eb_gpu = nullptr;
    T *dU, *dV, *dEb_U, *dEb_V;
    const T threshold = (T)(1.0 / (1 << 20));
    cudaStream_t stream;
    cudaEvent_t a, b;
    dim3 blockSize_v2(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 gridSize_v2((r2 + (blockSize_v2.x-2) - 1) / (blockSize_v2.x-2), 
        (r1 + (blockSize_v2.y-2)-1) / (blockSize_v2.y-2));
    dim3 blockSize_v3(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 gridSize_v3((r2 + (blockSize_v3.x-2) - 1) / (blockSize_v3.x-2), 
        (r1 + (blockSize_v3.y*NUM_PRE_THREAD-2)-1) / (blockSize_v3.y*NUM_PRE_THREAD-2));
    auto bytes = r1 * r2 * sizeof(T) * 2.0;
    auto GiB = 1024 * 1024 * 1024.0;
    int N = 1;
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + num_elements;
    T *ebIsZero_U_data = nullptr, *ebisZero_V_data = nullptr, *end_it = nullptr;
    uint32_t *data_indices = nullptr, *ebIsZero_U_indices = nullptr, *ebIsZero_V_indices = nullptr, *end_idx = nullptr;
    int zero_eb_U_count, zero_eb_V_count;
    float lrz_time = 0.0;
    float cusz_error_bound = 1e-5;
    float ms_derive_eb = 0.0f, ms_land = 0.0f, ms_ebzero = 0.0f;
    float ms_uni = 0.0f, ms_lrz = 0.0f, ms_tile_eb_pack = 0.0f;
    cudaEvent_t _ce0, _ce1;
    cudaEventCreate(&_ce0); cudaEventCreate(&_ce1);
    size_t hf_lens[4] = {0, 0, 0, 0};
    float hf_encode_ms[4] = {0, 0, 0, 0};
    float hf_decode_ms[4] = {0, 0, 0, 0};
    uint8_t* hf_blobs[4] = {nullptr, nullptr, nullptr, nullptr};

    Eq2* eq_dEb_V = nullptr;
    T *decomp_buf_U = nullptr, *decomp_buf_V = nullptr;
    uint8_t* d_land_bitpack = nullptr;
    size_t land_bitpack_bytes = 0;
    size_t land_bitpack_capacity = (num_elements + 7) / 8;
    size_t land_count = 0;

    cudaMalloc(&eb_gpu, r1 * r2 * sizeof(T));
    cudaMalloc(&dU, r1 * r2 * sizeof(T));
    cudaMalloc(&dV, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_U, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_V, r1 * r2 * sizeof(T));
    cudaMalloc(&eq_dEb_V, r1 * r2 * sizeof(Eq2));
    cudaMalloc(&decomp_buf_U, r1 * r2 * sizeof(T));
    cudaMalloc(&decomp_buf_V, r1 * r2 * sizeof(T));
    cudaMalloc(&data_indices, r1 * r2 * sizeof(uint32_t));
    cudaMalloc(&ebIsZero_U_data, r1 * r2 * sizeof(T));
    cudaMalloc(&ebIsZero_U_indices, r1 * r2 * sizeof(uint32_t));
    cudaMalloc(&ebisZero_V_data, r1 * r2 * sizeof(T));
    cudaMalloc(&ebIsZero_V_indices, r1 * r2 * sizeof(uint32_t));
    // Device counters for fused eb_zero kernel (reuse ot_num layout)
    uint32_t *d_zeb_U_cnt, *d_zeb_V_cnt;
    cudaMalloc(&d_zeb_U_cnt, sizeof(uint32_t));
    cudaMalloc(&d_zeb_V_cnt, sizeof(uint32_t));

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
        cudaMemset(eb_gpu, max_pwr_eb, r2 * r1 * sizeof(T));
        printf("gridSize_v2: %d, %d\n", gridSize_v2.x, gridSize_v2.y);
        cudaEventRecord(_ce0);
        derive_eb_offline_v2<<<gridSize_v2, blockSize_v2>>>(dU, dV, eb_gpu, dEb_U, dEb_V, eq_dEb, eq_dEb_V, r1, r2, max_pwr_eb);
        cudaEventRecord(_ce1);
        cudaEventSynchronize(_ce1);
        cudaEventElapsedTime(&ms_derive_eb, _ce0, _ce1);
        printf("compute V3 eb_gpu done\n");
        printf("\n");
        
        if(debug_options.verify_derive_eb){
            verify_derive_eb_cpu_vs_gpu(U, V, dEb_U, dEb_V, r1, r2, max_pwr_eb, threshold);
        }

        //DEAL_WITH_LAND
        if(debug_options.deal_with_land_data){
            //If U and V are both 0, means it is land in occean data, 
            //so it compress and decompress is all 0, we can just set eb = max-pwr
            printf("DEAL_WITH_LAND_DATA\n");
            // Precompute on CPU to avoid per-thread log2 on GPU
            const int land_id = (int)(log2(max_pwr_eb / threshold) / 2.0);
            const T   land_eb = (T)(1ULL << (2 * land_id)) * threshold;
            cudaEventRecord(_ce0);
            thrust::for_each(
                idx_first, idx_last,
                [=] __device__ (size_t i) {
                    if (dU[i] == (T)0 && dV[i] == (T)0) {
                        eq_dEb[i]   = (Eq2)land_id;
                        eq_dEb_V[i] = (Eq2)land_id;
                        dEb_U[i] = land_eb;
                        dEb_V[i] = land_eb;
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
            T eb_back = (T)(1ULL << (2 * id)) * threshold;
            cudaMemset(d_zeb_U_cnt, 0, sizeof(uint32_t));
            cudaMemset(d_zeb_V_cnt, 0, sizeof(uint32_t));
            cudaEventRecord(_ce0);
            constexpr int ZEB_BLK = 512;
            int zeb_grid = ((int)num_elements + ZEB_BLK - 1) / ZEB_BLK;
            kernel_fused_ebzero<T, Eq2><<<zeb_grid, ZEB_BLK>>>(
                dU, dV, dEb_U, dEb_V, eq_dEb, eq_dEb_V,
                ebIsZero_U_data, ebIsZero_U_indices, d_zeb_U_cnt,
                ebisZero_V_data, ebIsZero_V_indices, d_zeb_V_cnt,
                eb_back, threshold, (Eq2)id, num_elements);
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_ebzero, _ce0, _ce1);
            cudaMemcpy(&zero_eb_U_count, d_zeb_U_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&zero_eb_V_count, d_zeb_V_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("zero_eb_U_count: %d\n", zero_eb_U_count);
            printf("zero_eb_V_count: %d\n", zero_eb_V_count);
            printf("\n");
        }

        // TILE_UNIFORM_EB: broadcast per-tile min eb so the parallel
        // prefix-sum tile Lorenzo has consistent quantization units.
        if(debug_options.use_tile_uniform_eb){
            printf("TILE_UNIFORM_EB (tile=%dx%d)\n", LORENZO_TILE_DIM, LORENZO_TILE_DIM);
            dim3 ublk(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1);
            dim3 ugrid((r2 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM,
                       (r1 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM, 1);
            cudaEventRecord(_ce0);
            kernel_uniformize_tile_eb<T, Eq2><<<ugrid, ublk>>>(dEb_U, eq_dEb,   (int)r1, (int)r2, threshold);
            kernel_uniformize_tile_eb<T, Eq2><<<ugrid, ublk>>>(dEb_V, eq_dEb_V, (int)r1, (int)r2, threshold);
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_uni, _ce0, _ce1);
            printf("  uniformize eb: %.3f ms\n\n", ms_uni);
        }

        if(debug_options.run_quantization_lorenzo){
            printf("RUN_QUANTIZATION_LORENZO\n");
            cudaMemset(ot_num_U, 0, sizeof(uint32_t));
            cudaMemset(ot_num_V, 0, sizeof(uint32_t));
            *h_ot_num_U = 0, *h_ot_num_V=0;
            cudaMemset(eq_U, 0, r2 * r1 * sizeof(Eq1));

            if(debug_options.run_hf){
                // Free large arrays no longer needed before HF (saves ~6GB for large grids)
                cudaFree(eb_gpu);          eb_gpu = nullptr;
                cudaFree(data_indices);    data_indices = nullptr;
                cudaFree(decomp_buf_U);    decomp_buf_U = nullptr;
                cudaFree(decomp_buf_V);    decomp_buf_V = nullptr;
            }

            cudaEventRecord(_ce0);
            if (debug_options.use_tile_uniform_eb) {
                // Parallel prefix-sum tile Lorenzo (uniform eb per tile).
                // Outliers stored as delta-codes (candidate=delta+radius), pre-scatter convention.
                psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, Eq1>(
                    dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U,
                    dEb_U, RADIUS, &lrz_time, 0);
                cudaMemset(eq_V, 0, r2 * r1 * sizeof(Eq1));
                psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, Eq1>(
                    dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V,
                    dEb_V, RADIUS, &lrz_time, 0);
            } else {
                // Sequential row1d Lorenzo (exact-value outliers, post-scatter convention).
                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                    dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U,
                    dEb_U, RADIUS, &lrz_time, 0);
                cudaMemset(eq_V, 0, r2 * r1 * sizeof(Eq1));
                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                    dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V,
                    dEb_V, RADIUS, &lrz_time, 0);
            }
            cudaEventRecord(_ce1);
            cudaEventSynchronize(_ce1);
            cudaEventElapsedTime(&ms_lrz, _ce0, _ce1);
            cudaDeviceSynchronize();  // ensure ot_num arrays are ready for memcpy
            cudaMemcpy(h_ot_num_U, ot_num_U, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ot_num_V, ot_num_V, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("ot_num_U: %d, ot_num_V: %d\n", *h_ot_num_U, *h_ot_num_V);
            // Debug: check eq_U at known zero-eb position
            {
                const size_t dbg_pos = 218909;
                uint16_t eq_at_pos;
                cudaMemcpy(&eq_at_pos, (uint16_t*)eq_U + dbg_pos, sizeof(uint16_t), cudaMemcpyDeviceToHost);
                printf("  [dbg-compress-eq] eq_U[%zu]=%u (RADIUS=%d)\n", dbg_pos, eq_at_pos, RADIUS);
            }
            printf("\n");

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
                    run_gpu_huffman_u2_arrays(
                        reinterpret_cast<uint16_t*>(eq_U), reinterpret_cast<uint16_t*>(eq_V),
                        num_elements, stream,
                        eq_hf_lens, eq_hf_encode_ms, eq_hf_decode_ms,
                        eq_hf_blobs);
                    hf_lens[0] = eq_hf_lens[0];
                    hf_lens[1] = eq_hf_lens[1];
                    hf_encode_ms[0] = eq_hf_encode_ms[0];
                    hf_encode_ms[1] = eq_hf_encode_ms[1];
                    hf_decode_ms[0] = eq_hf_decode_ms[0];
                    hf_decode_ms[1] = eq_hf_decode_ms[1];
                    hf_blobs[0] = eq_hf_blobs[0];
                    hf_blobs[1] = eq_hf_blobs[1];

                    pack_tile_eb_payloads<Eq2>(
                        eq_dEb, eq_dEb_V, r1, r2,
                        hf_blobs[2], hf_blobs[3],
                        hf_lens[2], hf_lens[3],
                        ms_tile_eb_pack);

                    printf("  after  HF/pack: eq_U=%zu (%.1f%%)  eq_V=%zu (%.1f%%)  tile_dEb_U=%zu  tile_dEb_V=%zu\n",
                           hf_lens[0], hf_lens[0]*100.0/raw_eq,
                           hf_lens[1], hf_lens[1]*100.0/raw_eq,
                           hf_lens[2], hf_lens[3]);
                    printf("  tile EB payload: %zu tiles/field, %.3f ms pack\n",
                           hf_lens[2] / sizeof(Eq2), ms_tile_eb_pack);
                }
                else if (sizeof(Eq1) == sizeof(uint16_t) && sizeof(Eq2) == sizeof(uint8_t)) {
                    run_gpu_huffman_u2_u1_arrays(
                        reinterpret_cast<uint16_t*>(eq_U), reinterpret_cast<uint16_t*>(eq_V),
                        reinterpret_cast<uint8_t*>(eq_dEb), reinterpret_cast<uint8_t*>(eq_dEb_V),
                        num_elements, stream,
                        hf_lens, hf_encode_ms, hf_decode_ms,
                        hf_blobs);
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
                    float hf_ms = hf_encode_ms[0]+hf_encode_ms[1]+hf_encode_ms[2]+hf_encode_ms[3];
                    float total_ms = ms_derive_eb + ms_land + ms_ebzero + ms_uni + ms_lrz + hf_ms + ms_tile_eb_pack;
                    printf("COMPRESS_TIME\n");
                    printf("  derive_eb : %8.3f ms\n", ms_derive_eb);
                    printf("  land_data : %8.3f ms\n", ms_land);
                    printf("  eb_zero   : %8.3f ms\n", ms_ebzero);
                    printf("  uniform_eb: %8.3f ms\n", ms_uni);
                    printf("  lorenzo   : %8.3f ms\n", ms_lrz);
                    printf("  huffman   : %8.3f ms\n", hf_ms);
                    printf("  tile_eb   : %8.3f ms\n", ms_tile_eb_pack);
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
                    uint32_t* h_zeb_idx_U = new uint32_t[cnt_zeb_U ? cnt_zeb_U : 1];
                    float*    h_zeb_val_U = new float   [cnt_zeb_U ? cnt_zeb_U : 1];
                    uint32_t* h_ot_idx_V  = new uint32_t[cnt_ot_V  ? cnt_ot_V  : 1];
                    float*    h_ot_val_V  = new float   [cnt_ot_V  ? cnt_ot_V  : 1];
                    uint32_t* h_zeb_idx_V = new uint32_t[cnt_zeb_V ? cnt_zeb_V : 1];
                    float*    h_zeb_val_V = new float   [cnt_zeb_V ? cnt_zeb_V : 1];

                    if (cnt_ot_U) {
                        cudaMemcpy(h_ot_idx_U, ot_idx_U, cnt_ot_U * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_ot_val_U, ot_val_U, cnt_ot_U * sizeof(float),    cudaMemcpyDeviceToHost);
                    }
                    if (cnt_zeb_U) {
                        cudaMemcpy(h_zeb_idx_U, ebIsZero_U_indices, cnt_zeb_U * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_zeb_val_U, ebIsZero_U_data,    cnt_zeb_U * sizeof(float),    cudaMemcpyDeviceToHost);
                    }
                    if (cnt_ot_V) {
                        cudaMemcpy(h_ot_idx_V, ot_idx_V, cnt_ot_V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_ot_val_V, ot_val_V, cnt_ot_V * sizeof(float),    cudaMemcpyDeviceToHost);
                    }
                    if (cnt_zeb_V) {
                        cudaMemcpy(h_zeb_idx_V, ebIsZero_V_indices, cnt_zeb_V * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_zeb_val_V, ebisZero_V_data,    cnt_zeb_V * sizeof(float),    cudaMemcpyDeviceToHost);
                    }

                    uint8_t* h_land_bitpack = nullptr;
                    if (land_bitpack_bytes > 0) {
                        h_land_bitpack = new uint8_t[land_bitpack_bytes];
                        cudaMemcpy(h_land_bitpack, d_land_bitpack, land_bitpack_bytes, cudaMemcpyDeviceToHost);
                    }

                    write_cucpsz(out_fname, r1, r2, max_pwr_eb,
                        hf_blobs, hf_lens,
                        cnt_ot_U,  h_ot_idx_U,  h_ot_val_U,
                        cnt_zeb_U, h_zeb_idx_U, h_zeb_val_U,
                        cnt_ot_V,  h_ot_idx_V,  h_ot_val_V,
                        cnt_zeb_V, h_zeb_idx_V, h_zeb_val_V,
                        h_land_bitpack, land_bitpack_bytes,
                        debug_options.use_tile_uniform_eb ? 1u : 0u,
                        debug_options.use_tile_uniform_eb ? (uint32_t)LORENZO_TILE_DIM : 0u);
                    delete[] h_land_bitpack;

                    delete[] h_ot_idx_U; delete[] h_ot_val_U;
                    delete[] h_zeb_idx_U; delete[] h_zeb_val_U;
                    delete[] h_ot_idx_V; delete[] h_ot_val_V;
                    delete[] h_zeb_idx_V; delete[] h_zeb_val_V;
                    for (int i = 0; i < 4; i++) { delete[] hf_blobs[i]; hf_blobs[i] = nullptr; }
                }
            }

            if(debug_options.compute_ratio){
                printf("COMPUTE_RATIO\n");
                size_t sp_U_bytes = (*h_ot_num_U + (uint32_t)zero_eb_U_count) * (sizeof(uint32_t) + sizeof(float));
                size_t sp_V_bytes = (*h_ot_num_V + (uint32_t)zero_eb_V_count) * (sizeof(uint32_t) + sizeof(float));
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

            if(debug_options.write_data){
                printf("WRITE_DATA\n");
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

            //OVERALL_PERFORMANCE
            if(debug_options.overall_performance){
                printf("OVERALL_PERFORMANCE\n");
                    //Test end to end Compression
                    cudaStreamCreate(&stream);
                    cudaEventCreate(&a), cudaEventCreate(&b);
                    auto exec = thrust::cuda::par.on(stream);
                    for (int i_count=0;i_count<1;i_count++){
                        float ms = 0.0;
                        for (size_t i = 0; i < N; i++)
                        {
                            float temp;
                            cudaEventRecord(a, stream);
                            {
                                derive_eb_offline_v2<<<gridSize_v2, blockSize_v2, 0, stream>>>(dU, dV, eb_gpu, dEb_U, dEb_V, eq_dEb, eq_dEb_V, r1, r2, max_pwr_eb);
                                //derive_eb_offline_v3<<<gridSize_v3, blockSize_v3, 0, stream>>>(dU, dV, eb_gpu, dEb_U, dEb_V, eq_dEb, eq_dEb, r1, r2, max_pwr_eb);
                                thrust::for_each(
                                    exec, idx_first, idx_last,
                                    [=] __device__ (size_t i) {
                                        if (dU[i] == 0 && dV[i] == 0) {
                                            int id = log2(max_pwr_eb / threshold)/2.0;
                                            eq_dEb[i] = id;
                                            eq_dEb_V[i] = id;
                                            dEb_U[i] = (T)(1ULL << (2 * id)) * threshold;
                                            dEb_V[i] = (T)(1ULL << (2 * id)) * threshold;
                                        }
                                    }
                                );
                                thrust::sequence(exec, data_indices, data_indices + r1*r2);
                                int id = log2(max_pwr_eb / threshold) / 2.0;
                                T eb_back = (T)(1ULL << (2 * id)) * threshold;
                                //U
                                end_it = thrust::copy_if(exec, dU, dU + r1*r2, eq_dEb, ebIsZero_U_data, IsZero<T>());
                                end_idx = thrust::copy_if(exec, data_indices, data_indices + r1*r2, eq_dEb, ebIsZero_U_indices, IsZero<T>());
                                zero_eb_U_count = end_it - ebIsZero_U_data;
                                thrust::transform(exec, dEb_U, dEb_U + r1*r2, dEb_U, ReplaceLessThreshold(eb_back, threshold));
                                thrust::transform(exec, eq_dEb, eq_dEb + r1*r2, eq_dEb, ReplaceZero(id));
                                //V
                                end_it = thrust::copy_if(exec, dV, dV + r1*r2, eq_dEb_V, ebisZero_V_data, IsZero<T>());
                                end_idx = thrust::copy_if(exec, data_indices, data_indices + r1*r2, eq_dEb_V, ebIsZero_V_indices, IsZero<T>());
                                zero_eb_V_count = end_it - ebisZero_V_data;
                                thrust::transform(exec, dEb_V, dEb_V + r1*r2, dEb_V, ReplaceLessThreshold(eb_back, threshold));
                                thrust::transform(exec, eq_dEb_V, eq_dEb_V + r1*r2, eq_dEb_V, ReplaceZero(id));
                                cudaMemsetAsync(ot_num_U, 0, sizeof(uint32_t), stream);
                                cudaMemset(eq_U, 0, r2 * r1 * sizeof(Eq1));
                                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                                    dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U,
                                    dEb_U, RADIUS, &lrz_time, stream);
                                cudaMemsetAsync(ot_num_V, 0, sizeof(uint32_t), stream);
                                cudaMemset(eq_V, 0, r2 * r1 * sizeof(Eq1));
                                psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(
                                    dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V,
                                    dEb_V, RADIUS, &lrz_time, stream);
                            }
                            cudaEventRecord(b, stream);
                            cudaStreamSynchronize(stream);
                            cudaEventElapsedTime(&temp, a, b);
                            ms+=temp;
                        }
                        float huffman_time = hf_encode_ms[0] + hf_encode_ms[1] + hf_encode_ms[2] + hf_encode_ms[3];
                        printf("Overall time is %f ms, V3 speed GiB/s: %f\n", ms/N + huffman_time, bytes / GiB / ((ms / N + huffman_time) / 1000));
                    }
                    cudaStreamDestroy(stream);
                    printf("\n");
            }

            printf("\n");
        }
    }  

    if(debug_options.test_rightness){
        verify_decompressed_error_bounds(dEb_U, dEb_V, eq_dEb, eq_dEb_V, r1, r2, max_pwr_eb, threshold);
    }

    //decompression deb_U, deb_V
    T* dEb_U_dcomp; cudaMalloc(&dEb_U_dcomp, r1 * r2 * sizeof(T));
    T* dEb_V_dcomp; cudaMalloc(&dEb_V_dcomp, r1 * r2 * sizeof(T));
    thrust::for_each(
        thrust::device, idx_first, idx_last,
        [=] __device__ (size_t i) {
            dEb_U_dcomp[i] = (eq_dEb[i]   == 0) ? (T)0 : (T)(1ULL << (2 * eq_dEb[i]))   * threshold;
            dEb_V_dcomp[i] = (eq_dEb_V[i] == 0) ? (T)0 : (T)(1ULL << (2 * eq_dEb_V[i])) * threshold;
        }
    );


    //decompression U (correct float-space Lorenzo, outliers pre-scattered as exact values)
    T* dU_decomp; cudaMalloc(&dU_decomp, r1 * r2 * sizeof(T));
    cudaMemset(dU_decomp, 0, r1 * r2 * sizeof(T));
    cudaMemcpy(h_ot_num_U, ot_num_U, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (*h_ot_num_U)
        thrust::scatter(thrust::device, ot_val_U, ot_val_U + *h_ot_num_U, ot_idx_U, dU_decomp);
    if (debug_options.use_tile_uniform_eb)
        psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
            eq_U, dU_decomp, dU_decomp, dim3(r2, r1, 1), dEb_U_dcomp, RADIUS, &lrz_time, 0);
    else
        psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
            eq_U, dU_decomp, dim3(r2, r1, 1), dEb_U_dcomp, RADIUS, &lrz_time, 0);
    cudaDeviceSynchronize();
    thrust::scatter(thrust::device, ebIsZero_U_data, ebIsZero_U_data + zero_eb_U_count, ebIsZero_U_indices, dU_decomp);
    //decompression V
    T* dV_decomp; cudaMalloc(&dV_decomp, r1 * r2 * sizeof(T));
    cudaMemset(dV_decomp, 0, r1 * r2 * sizeof(T));
    cudaMemcpy(h_ot_num_V, ot_num_V, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (*h_ot_num_V)
        thrust::scatter(thrust::device, ot_val_V, ot_val_V + *h_ot_num_V, ot_idx_V, dV_decomp);
    if (debug_options.use_tile_uniform_eb)
        psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
            eq_V, dV_decomp, dV_decomp, dim3(r2, r1, 1), dEb_V_dcomp, RADIUS, &lrz_time, 0);
    else
        psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
            eq_V, dV_decomp, dim3(r2, r1, 1), dEb_V_dcomp, RADIUS, &lrz_time, 0);
    cudaDeviceSynchronize();
    thrust::scatter(thrust::device, ebisZero_V_data, ebisZero_V_data + zero_eb_V_count, ebIsZero_V_indices, dV_decomp);
    // Restore land cells (U=V=0) to exact zero using bitpack
    if (land_bitpack_bytes > 0 && d_land_bitpack) {
        uint8_t* bp = d_land_bitpack;
        T* dU_d = dU_decomp; T* dV_d = dV_decomp;
        thrust::for_each(idx_first, idx_last, [=] __device__ (size_t i) {
            if (bp[i/8] & (1u << (i%8))) { dU_d[i] = (T)0; dV_d[i] = (T)0; }
        });
    }

    //test decompression U,V
    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    auto exec = thrust::cuda::par.on(stream);
    for (int i_count=0;i_count<1;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaMemsetAsync(dU_decomp, 0, r1 * r2 * sizeof(T), stream);
            cudaMemsetAsync(dV_decomp, 0, r1 * r2 * sizeof(T), stream);
            cudaEventRecord(a, stream);
            thrust::for_each(
                exec, idx_first, idx_last,
                [=] __device__ (size_t i) {
                    dEb_U_dcomp[i] = (eq_dEb[i]   == 0) ? (T)0 : (T)(1ULL << (2 * eq_dEb[i]))   * threshold;
                    dEb_V_dcomp[i] = (eq_dEb_V[i] == 0) ? (T)0 : (T)(1ULL << (2 * eq_dEb_V[i])) * threshold;
                }
            );
            if (*h_ot_num_U) thrust::scatter(exec, ot_val_U, ot_val_U + *h_ot_num_U, ot_idx_U, dU_decomp);
            if (debug_options.use_tile_uniform_eb)
                psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
                    eq_U, dU_decomp, dU_decomp, dim3(r2, r1, 1), dEb_U_dcomp, RADIUS, &lrz_time, stream);
            else
                psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
                    eq_U, dU_decomp, dim3(r2, r1, 1), dEb_U_dcomp, RADIUS, &lrz_time, stream);
            thrust::scatter(exec, ebIsZero_U_data, ebIsZero_U_data + zero_eb_U_count, ebIsZero_U_indices, dU_decomp);
            if (*h_ot_num_V) thrust::scatter(exec, ot_val_V, ot_val_V + *h_ot_num_V, ot_idx_V, dV_decomp);
            if (debug_options.use_tile_uniform_eb)
                psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
                    eq_V, dV_decomp, dV_decomp, dim3(r2, r1, 1), dEb_V_dcomp, RADIUS, &lrz_time, stream);
            else
                psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
                    eq_V, dV_decomp, dim3(r2, r1, 1), dEb_V_dcomp, RADIUS, &lrz_time, stream);
            thrust::scatter(exec, ebisZero_V_data, ebisZero_V_data + zero_eb_V_count, ebIsZero_V_indices, dV_decomp);
            // Restore land cells
            if (land_bitpack_bytes > 0 && d_land_bitpack) {
              uint8_t* bp = d_land_bitpack; T* dU_d=dU_decomp; T* dV_d=dV_decomp;
              thrust::for_each(exec, idx_first, idx_last, [=] __device__ (size_t i) {
                  if (bp[i/8] & (1u<<(i%8))) { dU_d[i]=(T)0; dV_d[i]=(T)0; }
              });
            }
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        float huffman_time = hf_decode_ms[0] + hf_decode_ms[1] + hf_decode_ms[2] + hf_decode_ms[3];
        printf("Decompression U, V elasped time is %f ms, speed GiB/s: %f\n", ms/N + huffman_time, bytes / GiB / ((ms / N + huffman_time) / 1000));
    }
    cudaStreamDestroy(stream);

    cudaError_t err;
    printf("compute eq done\n");
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(eb_gpu);
        cudaFree(dU);
        cudaFree(dV);
        cudaFree(dU_decomp);
        cudaFree(dV_decomp);
        cudaFree(dEb_U);
        cudaFree(dEb_V);
        cudaFree(ebIsZero_U_data);
        cudaFree(ebIsZero_U_indices);
        cudaFree(ebisZero_V_data);
        cudaFree(ebIsZero_V_indices);
        cudaFree(data_indices);
        return 0;
    }

    //copy back
    cudaMemcpy(U_decomp,dU_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_decomp,dV_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    
    //cudaFree (some may have been freed early before HF)
    if (eb_gpu)       cudaFree(eb_gpu);
    cudaFree(dU);
    cudaFree(dV);
    cudaFree(dU_decomp);
    cudaFree(dV_decomp);
    cudaFree(dEb_U);
    cudaFree(dEb_V);
    cudaFree(eq_dEb_V);
    if (decomp_buf_U) cudaFree(decomp_buf_U);
    if (decomp_buf_V) cudaFree(decomp_buf_V);
    cudaFree(d_land_bitpack);
    cudaFree(ebIsZero_U_data);
    cudaFree(ebIsZero_U_indices);
    cudaFree(ebisZero_V_data);
    cudaFree(ebIsZero_V_indices);
    if (data_indices) cudaFree(data_indices);
    cudaFree(d_zeb_U_cnt);
    cudaFree(d_zeb_V_cnt);
    cudaEventDestroy(_ce0);
    cudaEventDestroy(_ce1);

    return 0;
}

template<typename T, typename Eq1 = uint16_t, typename Eq2 = uint8_t>
void sz_decompress_cp_preserve_2d_offline_gpu(
    const char* cucpsz_fname,
    T* U_decomp, T* V_decomp)
{
    FILE* f = fopen(cucpsz_fname, "rb");
    if (!f) { printf("Cannot open %s\n", cucpsz_fname); return; }
    CucpszHeader fhdr;
    fread(&fhdr, sizeof(fhdr), 1, f);
    size_t r1 = fhdr.r1, r2 = fhdr.r2, n = r1 * r2;
    const T threshold = (T)(1.0 / (1 << 20));
    bool tile_variant = (fhdr.lorenzo_variant == 1u);
    uint32_t file_tile_dim = fhdr.lorenzo_tile_dim ? fhdr.lorenzo_tile_dim : (uint32_t)LORENZO_TILE_DIM;
    printf("Decompress: r1=%zu r2=%zu  ot_U=%u zeb_U=%u  ot_V=%u zeb_V=%u\n",
           r1, r2, fhdr.ot_count_U, fhdr.zeroeb_count_U,
           fhdr.ot_count_V, fhdr.zeroeb_count_V);
    if (tile_variant)
        printf("  tile EB payload: tile=%u  U=%zu bytes  V=%zu bytes\n",
               file_tile_dim, (size_t)fhdr.hf_blob_len[2], (size_t)fhdr.hf_blob_len[3]);

    // Read 4 HF blobs: eq_U, eq_V, eq_dEb_U, eq_dEb_V
    uint8_t* h_blobs[4];
    for (int i = 0; i < 4; i++) {
        h_blobs[i] = new uint8_t[fhdr.hf_blob_len[i]];
        fread(h_blobs[i], 1, fhdr.hf_blob_len[i], f);
    }

    uint32_t cnt_ot_U  = fhdr.ot_count_U,  cnt_zeb_U = fhdr.zeroeb_count_U;
    uint32_t cnt_ot_V  = fhdr.ot_count_V,  cnt_zeb_V = fhdr.zeroeb_count_V;

    uint32_t* h_ot_idx_U  = new uint32_t[cnt_ot_U  ? cnt_ot_U  : 1];
    float*    h_ot_val_U  = new float   [cnt_ot_U  ? cnt_ot_U  : 1];
    uint32_t* h_zeb_idx_U = new uint32_t[cnt_zeb_U ? cnt_zeb_U : 1];
    float*    h_zeb_val_U = new float   [cnt_zeb_U ? cnt_zeb_U : 1];
    uint32_t* h_ot_idx_V  = new uint32_t[cnt_ot_V  ? cnt_ot_V  : 1];
    float*    h_ot_val_V  = new float   [cnt_ot_V  ? cnt_ot_V  : 1];
    uint32_t* h_zeb_idx_V = new uint32_t[cnt_zeb_V ? cnt_zeb_V : 1];
    float*    h_zeb_val_V = new float   [cnt_zeb_V ? cnt_zeb_V : 1];

    // Read land bitpack (comes before outlier/zeroeb arrays)
    size_t land_bp_bytes = fhdr.land_bitpack_bytes;
    uint8_t* h_land_bitpack = new uint8_t[land_bp_bytes ? land_bp_bytes : 1];
    if (land_bp_bytes) fread(h_land_bitpack, 1, land_bp_bytes, f);

    fread(h_ot_idx_U,  sizeof(uint32_t), cnt_ot_U,  f);
    fread(h_ot_val_U,  sizeof(float),    cnt_ot_U,  f);
    fread(h_zeb_idx_U, sizeof(uint32_t), cnt_zeb_U, f);
    fread(h_zeb_val_U, sizeof(float),    cnt_zeb_U, f);
    fread(h_ot_idx_V,  sizeof(uint32_t), cnt_ot_V,  f);
    fread(h_ot_val_V,  sizeof(float),    cnt_ot_V,  f);
    fread(h_zeb_idx_V, sizeof(uint32_t), cnt_zeb_V, f);
    fread(h_zeb_val_V, sizeof(float),    cnt_zeb_V, f);
    fclose(f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // HF decode eq arrays to GPU
    Eq1 *d_eq_U, *d_eq_V;
    Eq2 *d_eq_dEb_U = nullptr, *d_eq_dEb_V = nullptr;
    cudaMalloc(&d_eq_U,     n * sizeof(Eq1));
    cudaMalloc(&d_eq_V,     n * sizeof(Eq1));
    hf_u2_decode_from_blob(h_blobs[0], fhdr.hf_blob_len[0], n, d_eq_U,     stream);
    hf_u2_decode_from_blob(h_blobs[1], fhdr.hf_blob_len[1], n, d_eq_V,     stream);
    if (!tile_variant) {
        cudaMalloc(&d_eq_dEb_U, n * sizeof(Eq2));
        cudaMalloc(&d_eq_dEb_V, n * sizeof(Eq2));
        hf_u1_decode_from_blob(h_blobs[2], fhdr.hf_blob_len[2], n, d_eq_dEb_U, stream);
        hf_u1_decode_from_blob(h_blobs[3], fhdr.hf_blob_len[3], n, d_eq_dEb_V, stream);
    }

    // Upload outlier and zero-EB arrays to GPU
    uint32_t *d_ot_idx_U,  *d_zeb_idx_U;
    T        *d_ot_val_U,  *d_zeb_val_U;
    uint32_t *d_ot_idx_V,  *d_zeb_idx_V;
    T        *d_ot_val_V,  *d_zeb_val_V;
    cudaMalloc(&d_ot_idx_U,  (cnt_ot_U  ? cnt_ot_U  : 1) * sizeof(uint32_t));
    cudaMalloc(&d_ot_val_U,  (cnt_ot_U  ? cnt_ot_U  : 1) * sizeof(T));
    cudaMalloc(&d_zeb_idx_U, (cnt_zeb_U ? cnt_zeb_U : 1) * sizeof(uint32_t));
    cudaMalloc(&d_zeb_val_U, (cnt_zeb_U ? cnt_zeb_U : 1) * sizeof(T));
    cudaMalloc(&d_ot_idx_V,  (cnt_ot_V  ? cnt_ot_V  : 1) * sizeof(uint32_t));
    cudaMalloc(&d_ot_val_V,  (cnt_ot_V  ? cnt_ot_V  : 1) * sizeof(T));
    cudaMalloc(&d_zeb_idx_V, (cnt_zeb_V ? cnt_zeb_V : 1) * sizeof(uint32_t));
    cudaMalloc(&d_zeb_val_V, (cnt_zeb_V ? cnt_zeb_V : 1) * sizeof(T));
    if (cnt_ot_U) {
        cudaMemcpy(d_ot_idx_U, h_ot_idx_U, cnt_ot_U * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ot_val_U, h_ot_val_U, cnt_ot_U * sizeof(float),    cudaMemcpyHostToDevice);
    }
    if (cnt_zeb_U) {
        cudaMemcpy(d_zeb_idx_U, h_zeb_idx_U, cnt_zeb_U * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zeb_val_U, h_zeb_val_U, cnt_zeb_U * sizeof(float),    cudaMemcpyHostToDevice);
    }
    if (cnt_ot_V) {
        cudaMemcpy(d_ot_idx_V, h_ot_idx_V, cnt_ot_V * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ot_val_V, h_ot_val_V, cnt_ot_V * sizeof(float),    cudaMemcpyHostToDevice);
    }
    if (cnt_zeb_V) {
        cudaMemcpy(d_zeb_idx_V, h_zeb_idx_V, cnt_zeb_V * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zeb_val_V, h_zeb_val_V, cnt_zeb_V * sizeof(float),    cudaMemcpyHostToDevice);
    }

    // Reconstruct dEb arrays from separate eq_dEb_U and eq_dEb_V
    T *d_dEb_U, *d_dEb_V;
    cudaMalloc(&d_dEb_U, n * sizeof(T));
    cudaMalloc(&d_dEb_V, n * sizeof(T));
    if (tile_variant) {
        int tile_dim = (int)file_tile_dim;
        int tile_cols = ((int)r2 + tile_dim - 1) / tile_dim;
        int tile_rows = ((int)r1 + tile_dim - 1) / tile_dim;
        size_t tile_count = (size_t)tile_cols * tile_rows;
        size_t expected_tile_bytes = tile_count * sizeof(Eq2);
        if (fhdr.hf_blob_len[2] != expected_tile_bytes || fhdr.hf_blob_len[3] != expected_tile_bytes) {
            printf("  [warn] tile EB payload size mismatch: expected=%zu got U=%zu V=%zu\n",
                   expected_tile_bytes, (size_t)fhdr.hf_blob_len[2], (size_t)fhdr.hf_blob_len[3]);
        }

        Eq2 *d_tile_U = nullptr, *d_tile_V = nullptr;
        cudaMalloc(&d_tile_U, expected_tile_bytes);
        cudaMalloc(&d_tile_V, expected_tile_bytes);
        cudaMemcpy(d_tile_U, h_blobs[2], expected_tile_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tile_V, h_blobs[3], expected_tile_bytes, cudaMemcpyHostToDevice);

        dim3 block(16, 16, 1);
        dim3 grid(((int)r2 + block.x - 1) / block.x,
                  ((int)r1 + block.y - 1) / block.y, 1);
        kernel_expand_tile_eq_eb<T, Eq2><<<grid, block>>>(d_tile_U, d_dEb_U, (int)r1, (int)r2, tile_dim, tile_cols, threshold);
        kernel_expand_tile_eq_eb<T, Eq2><<<grid, block>>>(d_tile_V, d_dEb_V, (int)r1, (int)r2, tile_dim, tile_cols, threshold);
        cudaDeviceSynchronize();
        cudaFree(d_tile_U);
        cudaFree(d_tile_V);
    } else {
        thrust::counting_iterator<size_t> first(0);
        thrust::for_each(thrust::device, first, first + n, [=] __device__ (size_t i) {
            d_dEb_U[i] = (d_eq_dEb_U[i] == 0) ? (T)0 : (T)(1ULL << (2*d_eq_dEb_U[i])) * threshold;
            d_dEb_V[i] = (d_eq_dEb_V[i] == 0) ? (T)0 : (T)(1ULL << (2*d_eq_dEb_V[i])) * threshold;
        });
    }

    float lrz_time = 0.0f;

    // Decompression U (correct float-space Lorenzo):
    //   Pre-scatter exact original values at outlier positions.
    //   Sequential kernel reads pre-scattered values when qi==0 (no pred arithmetic).
    //   Post-scatter zero-EB exact floats.
    T* dU_decomp; cudaMalloc(&dU_decomp, n * sizeof(T));
    cudaMemset(dU_decomp, 0, n * sizeof(T));
    if (cnt_ot_U)
        thrust::scatter(thrust::device, d_ot_val_U, d_ot_val_U + cnt_ot_U, d_ot_idx_U, dU_decomp);
    if (tile_variant)
        psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
            d_eq_U, dU_decomp, dU_decomp, dim3(r2, r1, 1), d_dEb_U, RADIUS, &lrz_time, 0);
    else
        psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
            d_eq_U, dU_decomp, dim3(r2, r1, 1), d_dEb_U, RADIUS, &lrz_time, 0);
    cudaDeviceSynchronize();
    if (cnt_zeb_U)
        thrust::scatter(thrust::device, d_zeb_val_U, d_zeb_val_U + cnt_zeb_U, d_zeb_idx_U, dU_decomp);

    // Decompression V
    T* dV_decomp; cudaMalloc(&dV_decomp, n * sizeof(T));
    cudaMemset(dV_decomp, 0, n * sizeof(T));
    if (cnt_ot_V)
        thrust::scatter(thrust::device, d_ot_val_V, d_ot_val_V + cnt_ot_V, d_ot_idx_V, dV_decomp);
    if (tile_variant)
        psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, Eq1>(
            d_eq_V, dV_decomp, dV_decomp, dim3(r2, r1, 1), d_dEb_V, RADIUS, &lrz_time, 0);
    else
        psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<T, Eq1>(
            d_eq_V, dV_decomp, dim3(r2, r1, 1), d_dEb_V, RADIUS, &lrz_time, 0);
    cudaDeviceSynchronize();
    if (cnt_zeb_V)
        thrust::scatter(thrust::device, d_zeb_val_V, d_zeb_val_V + cnt_zeb_V, d_zeb_idx_V, dV_decomp);

    // Restore land cells (U=V=0) using bitpack
    if (land_bp_bytes > 0) {
        uint8_t* d_bp; cudaMalloc(&d_bp, land_bp_bytes);
        cudaMemcpy(d_bp, h_land_bitpack, land_bp_bytes, cudaMemcpyHostToDevice);
        T* dU_d = dU_decomp; T* dV_d = dV_decomp;
        thrust::counting_iterator<size_t> first2(0);
        thrust::for_each(thrust::device, first2, first2 + n, [=] __device__ (size_t i) {
            if (d_bp[i/8] & (1u << (i%8))) { dU_d[i] = (T)0; dV_d[i] = (T)0; }
        });
        cudaFree(d_bp);
    }

    cudaMemcpy(U_decomp, dU_decomp, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_decomp, dV_decomp, n * sizeof(T), cudaMemcpyDeviceToHost);

    // Cleanup GPU
    cudaFree(d_eq_U); cudaFree(d_eq_V);
    if (d_eq_dEb_U) cudaFree(d_eq_dEb_U);
    if (d_eq_dEb_V) cudaFree(d_eq_dEb_V);
    cudaFree(d_ot_idx_U); cudaFree(d_ot_val_U); cudaFree(d_zeb_idx_U); cudaFree(d_zeb_val_U);
    cudaFree(d_ot_idx_V); cudaFree(d_ot_val_V); cudaFree(d_zeb_idx_V); cudaFree(d_zeb_val_V);
    cudaFree(d_dEb_U); cudaFree(d_dEb_V);
    cudaFree(dU_decomp); cudaFree(dV_decomp);
    cudaStreamDestroy(stream);

    // Cleanup host
    for (int i = 0; i < 4; i++) delete[] h_blobs[i];
    delete[] h_land_bitpack;
    delete[] h_ot_idx_U; delete[] h_ot_val_U; delete[] h_zeb_idx_U; delete[] h_zeb_val_U;
    delete[] h_ot_idx_V; delete[] h_ot_val_V; delete[] h_zeb_idx_V; delete[] h_zeb_val_V;
}

// Simple verify: print max absolute error and count of violations > max_eb
template<typename T>
void verify(const T* orig, const T* decomp, size_t n) {
    double max_abs_err = 0.0, sum_sq = 0.0;
    double orig_max = orig[0], orig_min = orig[0];
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)orig[i] - (double)decomp[i]);
        if (err > max_abs_err) max_abs_err = err;
        sum_sq += err * err;
        if (orig[i] > orig_max) orig_max = orig[i];
        if (orig[i] < orig_min) orig_min = orig[i];
    }
    double rmse = sqrt(sum_sq / n);
    double range = orig_max - orig_min;
    double psnr = (max_abs_err > 0) ? 20.0 * log10(range / (2.0 * max_abs_err)) : 999.0;
    printf("  range=[%.4g, %.4g]  max_abs_err=%.6e  RMSE=%.6e  PSNR=%.2f dB\n",
           orig_min, orig_max, max_abs_err, rmse, psnr);
}

// Verify that two decompressed arrays are bit-identical (file I/O round-trip check)
template<typename T>
void verify_identical(const T* a, const T* b, size_t n, const char* label) {
    size_t diffs = 0;
    size_t first_diff = n;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (diffs == 0) first_diff = i;
            diffs++;
        }
    }
    if (diffs == 0) {
        printf("  %s: OK (bit-identical)\n", label);
    } else {
        printf("  %s: MISMATCH (%zu / %zu elements differ)\n", label, diffs, n);
        printf("    first diff at [%zu]: in-mem=%.8g  file=%.8g\n",
               first_diff, (double)a[first_diff], (double)b[first_diff]);
        // Print a few more differing indices
        size_t shown = 1;
        for (size_t i = first_diff + 1; i < n && shown < 5; i++) {
            if (a[i] != b[i]) {
                printf("    diff at [%zu]: in-mem=%.8g  file=%.8g\n",
                       i, (double)a[i], (double)b[i]);
                shown++;
            }
        }
    }
}

// ---- CP verification helpers ----

struct CriticalPoint2D {
    double x[2];
    int type;
    size_t simplex_id;
};

static void check_simplex_for_cp(
    const float* U, const float* V, int r2,
    int i0, int i1, int i2, size_t simplex_id,
    const double X[3][2],
    std::unordered_map<size_t, CriticalPoint2D>& cps)
{
    double v[3][2] = {
        {U[i0], V[i0]}, {U[i1], V[i1]}, {U[i2], V[i2]}
    };
    for (int k = 0; k < 3; k++)
        if (v[k][0] == 0 && v[k][1] == 0) return;
    double mu[3], cond;
    if (!ftk::inverse_lerp_s2v2(v, mu, &cond, 0.0)) return;
    double xp[2];
    ftk::lerp_s2v2(X, mu, xp);
    double J[2][2];
    ftk::jacobian_2dsimplex2(X, v, J);
    std::complex<double> eig[2];
    double delta = ftk::solve_eigenvalues2x2(J, eig);
    int type;
    if (delta >= 0) {
        if (eig[0].real() * eig[1].real() < 0)       type = ftk::CRITICAL_POINT_2D_SADDLE;
        else if (eig[0].real() < 0 && eig[1].real() < 0) type = ftk::CRITICAL_POINT_2D_ATTRACTING;
        else if (eig[0].real() > 0 && eig[1].real() > 0) type = ftk::CRITICAL_POINT_2D_REPELLING;
        else type = ftk::CRITICAL_POINT_2D_UNKNOWN;
    } else {
        if (eig[0].real() < 0)      type = ftk::CRITICAL_POINT_2D_ATTRACTING_FOCUS;
        else if (eig[0].real() > 0) type = ftk::CRITICAL_POINT_2D_REPELLING_FOCUS;
        else                         type = ftk::CRITICAL_POINT_2D_CENTER;
    }
    if (type == ftk::CRITICAL_POINT_2D_UNKNOWN) return;
    CriticalPoint2D cp;
    cp.x[0] = xp[0]; cp.x[1] = xp[1];
    cp.type = type; cp.simplex_id = simplex_id;
    cps[simplex_id] = cp;
}

static std::unordered_map<size_t, CriticalPoint2D>
compute_critical_points_2d(const float* U, const float* V, int r1, int r2)
{
    std::unordered_map<size_t, CriticalPoint2D> cps;
    const double X1[3][2] = {{0,0},{0,1},{1,1}};
    const double X2[3][2] = {{0,0},{1,0},{1,1}};
    for (int i = 1; i < r1-2; i++) {
        for (int j = 1; j < r2-2; j++) {
            size_t base = 2 * ((size_t)i * (r2-1) + j);
            int a=i*r2+j, b=(i+1)*r2+j, c=(i+1)*r2+(j+1), d=i*r2+(j+1);
            check_simplex_for_cp(U,V,r2, a,b,c, base,   X1, cps);
            check_simplex_for_cp(U,V,r2, a,d,c, base+1, X2, cps);
        }
    }
    return cps;
}

static void print_cp_comparison(
    const float* U_orig, const float* V_orig,
    const float* U_decomp, const float* V_decomp,
    int r1, int r2, const char* label)
{
    auto cps0 = compute_critical_points_2d(U_orig,   V_orig,   r1, r2);
    auto cps1 = compute_critical_points_2d(U_decomp, V_decomp, r1, r2);
    size_t tp = 0, fp = 0, fn = 0;
    for (auto& p : cps0)
        if (cps1.count(p.first)) tp++; else fn++;
    for (auto& p : cps1)
        if (!cps0.count(p.first)) fp++;
    printf("  CP [%s]: orig=%zu  decomp=%zu  TP=%zu  FP=%zu  FN=%zu\n",
           label, cps0.size(), cps1.size(), tp, fp, fn);
}

// ---- end CP verification helpers ----

int main(int argc, char ** argv){
    size_t num_elements = 0;
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    int r1 = atoi(argv[3]);
    int r2 = atoi(argv[4]);
    float max_eb = atof(argv[5]);

    cout << "start Compression\n";

    // float* ot_val_U; cudaMalloc(&ot_val_U, r2 * r1 * sizeof(float));
    // uint32_t* ot_idx_U; cudaMalloc(&ot_idx_U, r2 * r1 * sizeof(uint32_t));
    // uint32_t* ot_num_U; cudaMalloc(&ot_num_U, sizeof(uint32_t));
    // uint32_t* h_ot_num_U; cudaMallocHost(&h_ot_num_U, sizeof(uint32_t));
    // float* ot_val_V; cudaMalloc(&ot_val_V, r2 * r1 * sizeof(float));
    // uint32_t* ot_idx_V; cudaMalloc(&ot_idx_V, r2 * r1 * sizeof(uint32_t));
    // uint32_t* ot_num_V; cudaMalloc(&ot_num_V, sizeof(uint32_t));
    // uint32_t* h_ot_num_V; cudaMallocHost(&h_ot_num_V, sizeof(uint32_t))
    OtData ot_U;
    allocOtData(ot_U, num_elements);;
    OtData ot_V;
    allocOtData(ot_V, num_elements);
    uint16_t *eq_U, *eq_V;
    cudaMalloc(&eq_U, r2 * r1 * sizeof(uint16_t));
    cudaMalloc(&eq_V, r2 * r1 * sizeof(uint16_t));
    uint8_t *eq_dEb;
    cudaMalloc(&eq_dEb, r2 * r1 * sizeof(uint8_t));


    float * U_decomp = (float *) malloc(r1 * r2 * sizeof(float));
    float * V_decomp = (float *) malloc(r1 * r2 * sizeof(float));

    string cucpsz_fname = string(argv[1]) + ".cucpsz";

    unsigned char * result = sz_compress_cp_preserve_2d_offline_gpu(U, V,  eq_U, eq_V, eq_dEb, r1,  r2, max_eb,
        ot_U.val, ot_U.idx, ot_U.num, ot_U.h_num, ot_V.val, ot_V.idx, ot_V.num, ot_V.h_num, U_decomp, V_decomp,
        cucpsz_fname.c_str());

    // Decompress from file — this is the authoritative result
    printf("\nDecompressing from %s...\n", cucpsz_fname.c_str());
    float* U_decomp2 = new float[num_elements];
    float* V_decomp2 = new float[num_elements];
    sz_decompress_cp_preserve_2d_offline_gpu<float>(cucpsz_fname.c_str(), U_decomp2, V_decomp2);

    // Debug: compare in-memory vs file-based decompression
    verify_identical(U_decomp, U_decomp2, r1*(size_t)r2, "U in-mem vs file");
    verify_identical(V_decomp, V_decomp2, r1*(size_t)r2, "V in-mem vs file");

    printf("\nCP verification:\n");
    print_cp_comparison(U, V, U_decomp2, V_decomp2, r1, r2, "dataset");

    writefile((string(argv[1]) + ".out").c_str(), U_decomp2, num_elements);
    writefile((string(argv[2]) + ".out").c_str(), V_decomp2, num_elements);

    // Quality metrics: compare original vs file-decompressed
    printf("Quality (original vs decompressed):\n");
    printf("  U: "); verify(U, U_decomp2, num_elements);
    printf("  V: "); verify(V, V_decomp2, num_elements);

    delete[] U_decomp2;
    delete[] V_decomp2;

    free(result);
    free(U);
    free(V);
    free(U_decomp);
    free(V_decomp);
    freeOtData(ot_U);
    freeOtData(ot_V);
}
