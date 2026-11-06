#pragma once
#include <cstdint>

struct SmolLM2Config
{
    int32_t hidden_dim = 576;
    int32_t intermediate_size = 1536;
    int32_t n_layers = 30;
    int32_t n_heads = 9;
    int32_t n_kv_heads = 3;
    int32_t head_dim = 64;
    int32_t vocab_size = 49152;
    int32_t max_seq_len = 2048;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 100000.0f;
};

struct ModelHeader
{
    int32_t magic;
    int32_t vocab_size;
    int32_t dim;
    int32_t intermediate_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t max_seq_length;
    int32_t dtype_flag;   // 1 = FP16
    uint8_t padding[216]; // Total: exactly 256 bytes
};