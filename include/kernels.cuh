#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace kernels
{
    __global__ void get_embedding_vector(const int *tokens,             // [n_tokens]
                                         const __half *embedding_table, // [vocab_size * embedding_dim]
                                         __half *output,                // [n_tokens * embedding_dim]
                                         size_t n_tokens,
                                         int embedding_dim);

    __global__ void rmsnorm(const __half *x,      // [n_rows * d]
                            const __half *weight, // [d]
                            __half *out,          // [n_rows * d] (may alias x)
                            int d,
                            float epsilon = 1e-5f);

    __global__ void rope(__half *q,      // [seq_len * n_heads * head_dim]
                         __half *k,      // [seq_len * n_kv_heads * head_dim] (optional, may be nullptr)
                         int pos,        // Starting token position in sequence
                         int n_heads,
                         int n_kv_heads,
                         int head_dim,
                         float theta = 100000.0f);

    cublasStatus_t cublas_matmul(cublasHandle_t handle, __half *C, const __half *A, const __half *B, int M, int N, int K, __half alpha = 1.0f, __half beta = 0.0f);

    // Row-major linear projection: out [M, N] = x [M, K] * W^T where W is [N, K] (PyTorch Linear layout)
    cublasStatus_t linear_proj(cublasHandle_t handle, __half *out, const __half *x, const __half *w, int M, int N, int K);
    cublasStatus_t linear_proj_f32(cublasHandle_t handle, float *out, const __half *x, const __half *w, int M, int N, int K);

    __global__ void tiled_matmul(__half *C, const __half *A, const __half *B, int M, int N, int K);

    __global__ void softmax(const __half *x, // [n_rows * d]
                            __half *out,     // [n_rows * d] (may alias x)
                            int d);

    __global__ void append_kv_cache(__half *key_cache,
                                    __half *value_cache,
                                    const __half *k,
                                    const __half *v,
                                    int layer,
                                    int pos,
                                    int max_seq_len,
                                    int kv_dim);

    __global__ void gqa_attn_scores(const __half *q,
                                    const __half *key_cache,
                                    __half *attn_scores,
                                    int layer,
                                    int pos,
                                    int max_seq_len,
                                    int n_heads,
                                    int n_kv_heads,
                                    int head_dim);

    __global__ void gqa_attn_out(const __half *attn_scores,
                                 const __half *value_cache,
                                 __half *attn_out,
                                 int layer,
                                 int pos,
                                 int max_seq_len,
                                 int n_heads,
                                 int n_kv_heads,
                                 int head_dim);

    __global__ void swiglu(__half *gate, const __half *up, int d);

    __global__ void residual_add(__half *x, const __half *residual, int d);
}
