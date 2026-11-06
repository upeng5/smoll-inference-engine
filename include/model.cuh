#pragma once
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include "config.h"

// Pointers to weights inside the single allocated weights buffer
struct LayerWeights
{
    const __half *input_layernorm;          // [576]
    const __half *q_proj;                   // [576, 576]
    const __half *k_proj;                   // [192, 576]
    const __half *v_proj;                   // [192, 576]
    const __half *o_proj;                   // [576, 576]
    const __half *post_attention_layernorm; // [576]
    const __half *gate_proj;                // [1536, 576]
    const __half *up_proj;                  // [1536, 576]
    const __half *down_proj;                // [576, 1536]
};

// Reusable scratchpad memory for a single token's forward pass
struct RunState
{
    __half *x = nullptr;           // Current hidden state: [hidden_size]
    __half *xb = nullptr;          // Normalized hidden state / residual scratch: [hidden_size]
    __half *q = nullptr;           // Query buffer: [n_heads * head_dim] -> [576]
    __half *k = nullptr;           // Key buffer: [n_kv_heads * head_dim] -> [192]
    __half *v = nullptr;           // Value buffer: [n_kv_heads * head_dim] -> [192]
    __half *attn_scores = nullptr; // Attention logits: [n_heads * max_seq_len]
    __half *attn_out = nullptr;    // Attention sublayer output: [hidden_size]
    __half *mlp_gate = nullptr;    // SwiGLU gate intermediate: [intermediate_size] -> [1536]
    __half *mlp_up = nullptr;      // SwiGLU up intermediate: [intermediate_size] -> [1536]
    float *logits = nullptr;       // Logits (kept float32 for sampling): [vocab_size]

    // Key-Value Cache: [n_layers * max_seq_len * (n_kv_heads * head_dim)]
    __half *key_cache = nullptr;
    __half *value_cache = nullptr;

    // Host-side logits scratchpad for sampling
    std::vector<float> h_logits;

    void allocate(const SmolLM2Config &config);
    void free();
};

class SmolLM2Model
{
public:
    explicit SmolLM2Model(const SmolLM2Config &cfg) : config(cfg)
    {
        status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cuBLAS initialization failed" << std::endl;
            return;
        }
    }
    ~SmolLM2Model()
    {
        state.free();
        cudaFree(d_weights_raw);
        cublasDestroy(cublas_handle);
    }

    void load_model_weights(const std::string &file_path);
    void forward(int token, int pos, bool compute_logits = true);
    int sample(float temperature = 0.0f, float top_p = 0.9f);
    void generate();

    const SmolLM2Config &get_config() const { return config; }

    // Get output logits for token sampling
    const float *get_logits() const { return state.logits; }

private:
    SmolLM2Config config;

    // Single device allocation for all weights
    void *d_weights_raw = nullptr;

    // Global weights
    const __half *embed_tokens = nullptr; // [vocab_size, hidden_size]
    const __half *final_norm = nullptr;   // [hidden_size]
    const __half *lm_head = nullptr;      // Points to embed_tokens (tied)

    // Per-layer weight pointers
    std::vector<LayerWeights> layers;

    // Dynamic runtime scratchpads & KV cache
    RunState state;

    // Cuda variables
    cublasStatus_t status;
    cublasHandle_t cublas_handle;
};