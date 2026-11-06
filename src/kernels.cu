#include <cublas_v2.h>
#include "kernels.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#define TILE_WIDTH 16

// Launch: <<<n_tokens, embedding_dim>>> with block_size a power of two <= 1024 (e.g. 256).
// tokens: [120, 578, 477] = ["My", "Car", "Red"]
// Embedding vector index formula: token * embedding_dim
__global__ void kernels::get_embedding_vector(const int *tokens, const __half *embedding_table, __half *output, size_t n_tokens, int embedding_dim)
{
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    if (token_idx < n_tokens && dim_idx < embedding_dim)
    {
        int64_t vocab_id = tokens[token_idx];
        output[token_idx * embedding_dim + dim_idx] = embedding_table[vocab_id * embedding_dim + dim_idx];
    }
}

// Launch: <<<n_rows, block_size>>> with block_size a power of two <= 1024 (e.g. 256).
// Each block normalizes one row (one token's hidden vector) of length d.
// Safe to run in-place (out == x).
__global__ void kernels::rmsnorm(const __half *x, const __half *weight, __half *out, int d, float epsilon)
{
    __shared__ float partial[1024];

    const __half *row_in = x + (size_t)blockIdx.x * d;
    __half *row_out = out + (size_t)blockIdx.x * d;

    // 1. Each thread sums squares of its strided slice (accumulate in fp32 for accuracy)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        float v = __half2float(row_in[i]);
        sum_sq += v * v;
    }
    partial[threadIdx.x] = sum_sq;
    __syncthreads();

    // 2. Tree reduction in shared memory -> partial[0] holds the total
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    // 3. Every thread computes the same scale factor 1 / sqrt(mean + eps)
    float inv_rms = rsqrtf(partial[0] / d + epsilon);

    // 4. Normalize and apply the learned per-channel weight
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        row_out[i] = __float2half(__half2float(row_in[i]) * inv_rms * __half2float(weight[i]));
    }
}

// RoPE (Rotary Positional Embedding) — HuggingFace split-half layout:
// Pairs element i with element (i + head_dim / 2) within each head.
// Launch:
//   Single token: <<<n_heads, head_dim / 2>>>
//   Multi-token:  <<<dim3(n_heads, seq_len), head_dim / 2>>>
// Modifies q and k in-place.
__global__ void kernels::rope(__half *q, __half *k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta)
{
    int head_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int i = threadIdx.x;
    int half_dim = head_dim / 2;

    if (head_idx >= n_heads || i >= half_dim)
        return;

    // 1. Compute rotation angle for dimension pair i at token position (pos + token_idx)
    float freq = powf(theta, -(2.0f * i) / head_dim);
    float angle = (pos + token_idx) * freq;
    float sin_val, cos_val;
    sincosf(angle, &sin_val, &cos_val);

    // 2. Rotate Query head in-place
    __half *q_head = q + ((size_t)token_idx * n_heads + head_idx) * head_dim;
    float q0 = __half2float(q_head[i]);
    float q1 = __half2float(q_head[i + half_dim]);
    q_head[i]            = __float2half(q0 * cos_val - q1 * sin_val);
    q_head[i + half_dim] = __float2half(q1 * cos_val + q0 * sin_val);

    // 3. Rotate Key head in-place (GQA: only the first n_kv_heads blocks need to work)
    if (k != nullptr && head_idx < n_kv_heads)
    {
        __half *k_head = k + ((size_t)token_idx * n_kv_heads + head_idx) * head_dim;
        float k0 = __half2float(k_head[i]);
        float k1 = __half2float(k_head[i + half_dim]);
        k_head[i]            = __float2half(k0 * cos_val - k1 * sin_val);
        k_head[i + half_dim] = __float2half(k1 * cos_val + k0 * sin_val);
    }
}

// Row-major: C (MxN) = alpha * A (MxK) * B (KxN) + beta * C
cublasStatus_t kernels::cublas_matmul(cublasHandle_t handle, __half *C, const __half *A, const __half *B, int M, int N, int K, __half alpha, __half beta)
{
    // Row-major trick: C^T = B^T * A^T, evaluated in column-major
    return cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, // B^T: N x K col-major, ld = N
        A, K, // A^T: K x M col-major, ld = K
        &beta,
        C, N // C^T: N x M col-major, ld = N
    );
}

// Row-major linear projection: out [M, N] = x [M, K] * W^T where W is [N, K] row-major (FP32 compute, FP16 out)
cublasStatus_t kernels::linear_proj(cublasHandle_t handle, __half *out, const __half *x, const __half *w, int M, int N, int K)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    // In column-major: out^T [N, M] = (W^T_col [K, N])^T * x^T_col [K, M]
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w, CUDA_R_16F, K,
        x, CUDA_R_16F, K,
        &beta,
        out, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}

// Row-major linear projection with FP32 output (used for final logits): out [M, N] = x [M, K] * W^T
cublasStatus_t kernels::linear_proj_f32(cublasHandle_t handle, float *out, const __half *x, const __half *w, int M, int N, int K)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w, CUDA_R_16F, K,
        x, CUDA_R_16F, K,
        &beta,
        out, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}

__global__ void kernels::tiled_matmul(__half *C, const __half *A, const __half *B, int M, int N, int K)
{
    // Allocate shared memory for the current tiles of A and B
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // Identify row and column of the C element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __half pValue = 0.0f;

    // Loop over all the tiles required to compute the C element
    // Total number of phases = ceil(K / TILE_WIDTH)
    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
    {

        // 1. Collaborative loading of tile A into shared memory
        // Check boundary conditions for A (row must be < M, col in tile must be < K)
        if (row < M && (ph * TILE_WIDTH + threadIdx.x) < K)
        {
            ds_A[threadIdx.y][threadIdx.x] = A[row * K + ph * TILE_WIDTH + threadIdx.x];
        }
        else
        {
            ds_A[threadIdx.y][threadIdx.x] = 0.0f; // Pad with 0 if out of bounds
        }

        // 2. Collaborative loading of tile B into shared memory
        // Check boundary conditions for B (row in tile must be < K, col must be < N)
        if ((ph * TILE_WIDTH + threadIdx.y) < K && col < N)
        {
            ds_B[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * N + col];
        }
        else
        {
            ds_B[threadIdx.y][threadIdx.x] = 0.0f; // Pad with 0 if out of bounds
        }

        // Synchronize threads to ensure the entire tile is loaded before computing
        __syncthreads();

        // 3. Multiply the two tiles together
        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            pValue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }

        // Synchronize before moving to the next tile phase
        // Prevents threads from overwriting shared memory before others finish computing
        __syncthreads();
    }

    // Write the final accumulated value back to Global Memory C
    if (row < M && col < N)
    {
        C[row * N + col] = pValue;
    }
}

// Numerically stable Softmax: out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// Launch: <<<n_rows, block_size>>> with block_size a power of two <= 1024 (e.g. 256).
// Each block normalizes one row of length d.
// Safe to run in-place (out == x).
__global__ void kernels::softmax(const __half *x, __half *out, int d)
{
    __shared__ float partial[1024];

    const __half *row_in = x + (size_t)blockIdx.x * d;
    __half *row_out = out + (size_t)blockIdx.x * d;

    // 1. Each thread finds the max of its strided slice (for numerical stability)
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        float v = __half2float(row_in[i]);
        thread_max = fmaxf(thread_max, v);
    }
    partial[threadIdx.x] = thread_max;
    __syncthreads();

    // Tree reduction in shared memory -> partial[0] holds row max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = partial[0];
    __syncthreads();

    // 2. Each thread sums exp(x[i] - row_max) across its strided slice
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        sum_exp += expf(__half2float(row_in[i]) - row_max);
    }
    partial[threadIdx.x] = sum_exp;
    __syncthreads();

    // Tree reduction in shared memory -> partial[0] holds total sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / partial[0];

    // 3. Normalize to probabilities
    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        row_out[i] = __float2half(expf(__half2float(row_in[i]) - row_max) * inv_sum);
    }
}

__global__ void kernels::append_kv_cache(__half *key_cache, __half *value_cache, const __half *k, const __half *v, int layer, int pos, int max_seq_len, int kv_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim)
    {
        size_t offset = ((size_t)layer * max_seq_len + pos) * kv_dim + i;
        key_cache[offset] = k[i];
        value_cache[offset] = v[i];
    }
}

// Grouped-Query Attention (GQA) scaled dot-product scores for 1 query token at `pos` over [0 .. pos]
// Launch: <<<n_heads, 256>>>
__global__ void kernels::gqa_attn_scores(const __half *q, const __half *key_cache, __half *attn_scores, int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim)
{
    int h = blockIdx.x;
    if (h >= n_heads)
        return;

    int kv_mul = n_heads / n_kv_heads;
    int kv_h = h / kv_mul;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = pos + 1;
    float scale = rsqrtf((float)head_dim);

    const __half *q_head = q + h * head_dim;
    const __half *layer_k_cache = key_cache + (size_t)layer * max_seq_len * kv_dim;
    __half *head_scores = attn_scores + (size_t)h * seq_len;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)
    {
        const __half *k_head = layer_k_cache + (size_t)t * kv_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
        {
            dot += __half2float(q_head[d]) * __half2float(k_head[d]);
        }
        head_scores[t] = __float2half(dot * scale);
    }
}

// Grouped-Query Attention (GQA) weighted sum of V over [0 .. pos]
// Launch: <<<n_heads, head_dim>>>
__global__ void kernels::gqa_attn_out(const __half *attn_scores, const __half *value_cache, __half *attn_out, int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim)
        return;

    int kv_mul = n_heads / n_kv_heads;
    int kv_h = h / kv_mul;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = pos + 1;

    const __half *head_scores = attn_scores + (size_t)h * seq_len;
    const __half *layer_v_cache = value_cache + (size_t)layer * max_seq_len * kv_dim;

    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t)
    {
        const __half *v_head = layer_v_cache + (size_t)t * kv_dim + kv_h * head_dim;
        acc += __half2float(head_scores[t]) * __half2float(v_head[d]);
    }
    attn_out[h * head_dim + d] = __float2half(acc);
}

// SwiGLU activation: gate[i] = SiLU(gate[i]) * up[i]
__global__ void kernels::swiglu(__half *gate, const __half *up, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d)
    {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float silu = g / (1.0f + expf(-g));
        gate[i] = __float2half(silu * u);
    }
}

// In-place residual connection: x[i] += residual[i]
__global__ void kernels::residual_add(__half *x, const __half *residual, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d)
    {
        x[i] = __float2half(__half2float(x[i]) + __half2float(residual[i]));
    }
}
