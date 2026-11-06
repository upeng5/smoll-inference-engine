#include "model.cuh"
#include "kernels.cuh"
#include "profiling.h"
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

void RunState::allocate(const SmolLM2Config &cfg)
{
    free();

    const size_t dim = cfg.hidden_dim;
    const size_t kv_dim = (size_t)cfg.n_kv_heads * cfg.head_dim;
    const size_t inter_dim = cfg.intermediate_size;
    const size_t kv_cache_elems = (size_t)cfg.n_layers * cfg.max_seq_len * kv_dim;

    cudaMalloc(&x, dim * sizeof(__half));
    cudaMalloc(&xb, dim * sizeof(__half));
    cudaMalloc(&q, (size_t)cfg.n_heads * cfg.head_dim * sizeof(__half));
    cudaMalloc(&k, kv_dim * sizeof(__half));
    cudaMalloc(&v, kv_dim * sizeof(__half));
    cudaMalloc(&attn_scores, (size_t)cfg.n_heads * cfg.max_seq_len * sizeof(__half));
    cudaMalloc(&attn_out, dim * sizeof(__half));
    cudaMalloc(&mlp_gate, inter_dim * sizeof(__half));
    cudaMalloc(&mlp_up, inter_dim * sizeof(__half));
    cudaMalloc(&logits, (size_t)cfg.vocab_size * sizeof(float));

    cudaMalloc(&key_cache, kv_cache_elems * sizeof(__half));
    cudaMalloc(&value_cache, kv_cache_elems * sizeof(__half));

    cudaMemset(key_cache, 0, kv_cache_elems * sizeof(__half));
    cudaMemset(value_cache, 0, kv_cache_elems * sizeof(__half));

    h_logits.resize(cfg.vocab_size);
}

void RunState::free()
{
    cudaFree(x);           x = nullptr;
    cudaFree(xb);          xb = nullptr;
    cudaFree(q);           q = nullptr;
    cudaFree(k);           k = nullptr;
    cudaFree(v);           v = nullptr;
    cudaFree(attn_scores); attn_scores = nullptr;
    cudaFree(attn_out);    attn_out = nullptr;
    cudaFree(mlp_gate);    mlp_gate = nullptr;
    cudaFree(mlp_up);      mlp_up = nullptr;
    cudaFree(logits);      logits = nullptr;
    cudaFree(key_cache);   key_cache = nullptr;
    cudaFree(value_cache); value_cache = nullptr;
}

void SmolLM2Model::load_model_weights(const std::string &file_path)
{
    // Load header
    FILE *fptr = fopen(file_path.c_str(), "rb");
    if (!fptr)
    {
        std::cerr << "Could not open model file!\n";
        return;
    }

    ModelHeader h;
    if (fread(&h, sizeof(h), 1, fptr) != 1 || h.magic != 0x534D4F4C)
    {
        std::cerr << "Bad header!\n";
        fclose(fptr);
        return;
    }

    // Define constants used by offset
    // Global tensors
    const int embed_dim = h.vocab_size * h.dim;
    const int final_norm_dim = h.dim;

    // Per-layer offsets
    const int input_layernorm_dim = h.dim;

    const int mlp_down_proj_dim = h.dim * h.intermediate_dim;
    const int mlp_gate_proj_dim = h.intermediate_dim * h.dim;
    const int mlp_up_proj_dim = h.intermediate_dim * h.dim;

    const int post_attn_layernorm_dim = h.dim;

    const int attn_k_proj_dim = h.n_kv_heads * h.head_dim * h.dim;
    const int attn_o_proj_dim = h.dim * h.dim;
    const int attn_q_proj_dim = h.dim * h.dim;
    const int attn_v_proj_dim = h.n_kv_heads * h.head_dim * h.dim;

    // Calculate total global tensors and total accumulated memory for all layers
    const int global_count = (embed_dim + final_norm_dim);

    const int layer_mlp_count = mlp_down_proj_dim + mlp_gate_proj_dim + mlp_up_proj_dim;
    const int layer_attn_count = attn_k_proj_dim + attn_o_proj_dim + attn_q_proj_dim + attn_v_proj_dim;

    const int total_layers_count = (input_layernorm_dim + layer_mlp_count + post_attn_layernorm_dim + layer_attn_count) * h.n_layers;
    const int total_count = global_count + total_layers_count;

    // Load tensors into cpu
    std::vector<__half> tensors(total_count);
    if (fseek(fptr, sizeof(h), SEEK_SET) != 0)
    {
        std::cerr << "Failed to seek. \n";
        return;
    }

    if (fread(tensors.data(), sizeof(__half), total_count, fptr) != total_count)
    {
        std::cerr << "Couldn't read tensors from file!\n";
        fclose(fptr);
        return;
    }

    // Allocate global device memory
    cudaError_t status = cudaMalloc(&d_weights_raw, total_count * sizeof(__half));
    if (status != cudaSuccess)
    {
        std::cerr << "Couldn't allocate device memory. \n"
                  << cudaGetErrorString(status) << "\n"
                  << __FILE__ << " " << __LINE__ << "\n";
        fclose(fptr);
        cudaFree(d_weights_raw);
        d_weights_raw = nullptr;
        return;
    }

    // Copy tensor data into GPU
    status = cudaMemcpy(d_weights_raw, tensors.data(), total_count * sizeof(__half), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "Couldn't copy from host to device. \n"
                  << cudaGetErrorString(status) << "\n"
                  << __FILE__ << " " << __LINE__ << "\n";
        fclose(fptr);
        cudaFree(d_weights_raw);
        d_weights_raw = nullptr;
        return;
    }

    // Step 2: Calculate offsets for each layer
    const __half *base = static_cast<__half *>(d_weights_raw);
    size_t offset = 0;
    auto carve = [&](size_t count)
    {
        const __half *p = base + offset;
        offset += count;
        return p;
    };

    embed_tokens = carve(embed_dim);
    final_norm = carve(final_norm_dim);
    lm_head = embed_tokens;

    layers.resize(h.n_layers);
    for (int i = 0; i < h.n_layers; i++)
    {
        LayerWeights &L = layers[i];
        L.input_layernorm = carve(input_layernorm_dim);
        L.q_proj = carve(attn_q_proj_dim);
        L.k_proj = carve(attn_k_proj_dim);
        L.v_proj = carve(attn_v_proj_dim);
        L.o_proj = carve(attn_o_proj_dim);
        L.post_attention_layernorm = carve(post_attn_layernorm_dim);
        L.gate_proj = carve(mlp_gate_proj_dim);
        L.up_proj = carve(mlp_up_proj_dim);
        L.down_proj = carve(mlp_down_proj_dim);
    }

    if (offset != (size_t)total_count)
        std::cerr << "Offset mismatch: " << offset << " vs " << total_count << "\n";

    // Sync runtime config with loaded header and allocate RunState buffers
    config.vocab_size = h.vocab_size;
    config.hidden_dim = h.dim;
    config.intermediate_size = h.intermediate_dim;
    config.n_layers = h.n_layers;
    config.n_heads = h.n_heads;
    config.n_kv_heads = h.n_kv_heads;
    config.head_dim = h.head_dim;
    config.max_seq_len = h.max_seq_length;

    state.allocate(config);

    fclose(fptr);
}

void SmolLM2Model::forward(int token, int pos, bool compute_logits)
{
    const int dim = config.hidden_dim;
    const int kv_dim = config.n_kv_heads * config.head_dim;
    const int inter_dim = config.intermediate_size;
    const float eps = config.rms_norm_eps;

    NVTX_RANGE("forward");

    // 1. Copy token embedding into state.x
    {
        NVTX_RANGE("embedding");
        cudaMemcpy(state.x, embed_tokens + (size_t)token * dim, dim * sizeof(__half), cudaMemcpyDeviceToDevice);
    }

    // 2. Run all transformer decoder blocks
    for (int l = 0; l < config.n_layers; ++l)
    {
        NVTX_RANGE("layer");
        const LayerWeights &L = layers[l];

        NvtxRange attn_range("attention");
        // 2a. Pre-attention RMSNorm
        kernels::rmsnorm<<<1, 256>>>(state.x, L.input_layernorm, state.xb, dim, eps);

        // 2b. Q, K, V linear projections
        kernels::linear_proj(cublas_handle, state.q, state.xb, L.q_proj, 1, dim, dim);
        kernels::linear_proj(cublas_handle, state.k, state.xb, L.k_proj, 1, kv_dim, dim);
        kernels::linear_proj(cublas_handle, state.v, state.xb, L.v_proj, 1, kv_dim, dim);

        // 2c. Apply RoPE to Q and K in-place
        kernels::rope<<<config.n_heads, config.head_dim / 2>>>(
            state.q, state.k, pos, config.n_heads, config.n_kv_heads, config.head_dim, config.rope_theta);

        // 2d. Save K and V into KV cache at (l, pos)
        kernels::append_kv_cache<<<(kv_dim + 255) / 256, 256>>>(
            state.key_cache, state.value_cache, state.k, state.v, l, pos, config.max_seq_len, kv_dim);

        // 2e. GQA scaled dot-product attention scores over [0 .. pos]
        kernels::gqa_attn_scores<<<config.n_heads, 256>>>(
            state.q, state.key_cache, state.attn_scores, l, pos, config.max_seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim);

        // 2f. Softmax over attention scores
        kernels::softmax<<<config.n_heads, 256>>>(state.attn_scores, state.attn_scores, pos + 1);

        // 2g. GQA weighted sum of cached values
        kernels::gqa_attn_out<<<config.n_heads, config.head_dim>>>(
            state.attn_scores, state.value_cache, state.attn_out, l, pos, config.max_seq_len,
            config.n_heads, config.n_kv_heads, config.head_dim);

        // 2h. Output projection + residual connection
        kernels::linear_proj(cublas_handle, state.xb, state.attn_out, L.o_proj, 1, dim, dim);
        kernels::residual_add<<<(dim + 255) / 256, 256>>>(state.x, state.xb, dim);
        attn_range.end();
        NvtxRange mlp_range("mlp");

        // 2i. Post-attention RMSNorm
        kernels::rmsnorm<<<1, 256>>>(state.x, L.post_attention_layernorm, state.xb, dim, eps);

        // 2j. SwiGLU MLP + residual connection
        kernels::linear_proj(cublas_handle, state.mlp_gate, state.xb, L.gate_proj, 1, inter_dim, dim);
        kernels::linear_proj(cublas_handle, state.mlp_up, state.xb, L.up_proj, 1, inter_dim, dim);
        kernels::swiglu<<<(inter_dim + 255) / 256, 256>>>(state.mlp_gate, state.mlp_up, inter_dim);
        kernels::linear_proj(cublas_handle, state.xb, state.mlp_gate, L.down_proj, 1, dim, inter_dim);
        kernels::residual_add<<<(dim + 255) / 256, 256>>>(state.x, state.xb, dim);
    }

    // 3. Final RMSNorm & LM Head projection (only needed when sampling next token)
    if (compute_logits)
    {
        NVTX_RANGE("lm_head");
        kernels::rmsnorm<<<1, 256>>>(state.x, final_norm, state.xb, dim, eps);
        kernels::linear_proj_f32(cublas_handle, state.logits, state.xb, lm_head, 1, config.vocab_size, dim);
    }
}

int SmolLM2Model::sample(float temperature, float top_p)
{
    NVTX_RANGE("sample");
    cudaMemcpy(state.h_logits.data(), state.logits, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Greedy argmax when temperature <= 0
    if (temperature <= 0.0f)
    {
        auto it = std::max_element(state.h_logits.begin(), state.h_logits.end());
        return static_cast<int>(std::distance(state.h_logits.begin(), it));
    }

    // Temperature scaling + numerically stable softmax
    float max_logit = *std::max_element(state.h_logits.begin(), state.h_logits.end());
    float sum_probs = 0.0f;
    for (float &v : state.h_logits)
    {
        v = std::exp((v - max_logit) / temperature);
        sum_probs += v;
    }
    for (float &v : state.h_logits)
    {
        v /= sum_probs;
    }

    // Top-p (nucleus) sampling
    std::vector<int> indices(config.vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return state.h_logits[a] > state.h_logits[b]; });

    float cumulative = 0.0f;
    size_t cutoff = indices.size();
    for (size_t i = 0; i < indices.size(); ++i)
    {
        cumulative += state.h_logits[indices[i]];
        if (cumulative >= top_p)
        {
            cutoff = i + 1;
            break;
        }
    }

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, cumulative);
    float r = dist(rng);

    float acc = 0.0f;
    for (size_t i = 0; i < cutoff; ++i)
    {
        acc += state.h_logits[indices[i]];
        if (r <= acc)
            return indices[i];
    }
    return indices[cutoff - 1];
}