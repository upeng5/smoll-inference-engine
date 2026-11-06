# SmolLM2-135M CUDA Inference Engine

A C++17 and CUDA inference runtime for `HuggingFaceTB/SmolLM2-135M-Instruct`. It loads FP16 weights from a flat binary format into a single GPU allocation and runs autoregressive decoding with custom CUDA kernels and cuBLAS.

## Architecture and implementation

`SmolLM2-135M` is a 30-layer Llama-style decoder model (`d_model = 576`, `d_ff = 1536`, 9 query heads, 3 KV heads, `head_dim = 64`, `vocab_size = 49152`, `max_seq_len = 2048`, `rope_theta = 100000`).

### Memory layout

`scripts/helper.py` converts Hugging Face `.safetensors` weights into `dist/model.bin`: a 256-byte header (`magic = 0x534D4F4C`) followed by contiguous FP16 tensors. At startup, `SmolLM2Model::load_model_weights` allocates one device buffer (`~269 MB`) for all model parameters and assigns per-layer weight pointers by offset. The output projection (`lm_head`) aliases `embed_tokens` directly in device memory to support tied embeddings without duplicating the `[49152, 576]` table.

`RunState` pre-allocates all activation buffers and the FP16 KV cache (`[30, 2048, 192]` for K and V, `~45 MB` total) before generation starts, so the token loop performs zero `cudaMalloc` or `cudaFree` calls.

### Kernels and compute path

Linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`) call `cublasGemmEx` with `CUDA_R_16F` inputs and `CUBLAS_COMPUTE_32F` accumulation, transposing the `[N, K]` PyTorch row-major weight layout in place via column-major leading dimensions. The final `lm_head` projection writes `CUDA_R_32F` logits directly for sampling. During prefill, the engine skips `final_norm` and `lm_head` on all prompt tokens except the last one.

The remaining operations run in custom CUDA kernels in `src/kernels.cu`:

- `rmsnorm`: FP32 sum-of-squares accumulation with shared-memory tree reduction (`<<<1, 256>>>` per token).
- `rope`: In-place Rotary Positional Embeddings using Hugging Face's split-half pairing `(i, i + head_dim / 2)`, rotating Q and K in a single launch (`<<<n_heads, head_dim / 2>>>`).
- `append_kv_cache`: Writes the current step's K and V vectors into the layer's `[max_seq_len, kv_dim]` slice.
- `gqa_attn_scores`, `softmax`, `gqa_attn_out`: Grouped-Query Attention (3:1 query-to-KV head ratio) over `[0 .. pos]` with FP32 dot-product accumulation and shared-memory max and sum reductions.
- `swiglu` and `residual_add`: Elementwise SiLU gating and residual stream updates.

## Benchmark results

Measured with `run_bench` (`src/main.cu`) on a laptop NVIDIA GeForce MX450 (Turing `sm_75`, 14 SMs, 2 GB VRAM) using a 45-token ChatML prompt, 128 generated tokens, 2 warmup runs, and 5 measured runs:

```text
=== Benchmark (prompt=45 tokens, generated=128, runs=5, warmup=2) ===
prefill_ms_median:        288.29  (156.093 tok/s, sequential)
decode_e2e_tok_s_median:  116.462
token_ms_p50 / p95:       8.1396 / 10.0359
decode_forward_tok_s:     125.901  (no per-token host sync)
gpu_mem_in_use_mb:        795
```

| Metric | Value | Notes |
| :--- | :--- | :--- |
| Prefill latency (median) | `288.29 ms` (`156.1 tok/s`) | 45 prompt tokens processed sequentially (`lm_head` skipped until final token) |
| Decode throughput (end-to-end) | `116.5 tok/s` | Full forward pass + 196 KB `logits` D2H copy + CPU argmax per token |
| Per-token latency (p50 / p95) | `8.14 ms` / `10.04 ms` | Measured across 640 generated tokens (`5 runs x 128 tokens`) |
| Decode throughput (forward-only) | `125.9 tok/s` | Back-to-back GPU forward passes without per-token host synchronization |
| Total GPU memory in use | `795 MB` | Includes weights (`256 MB`), KV cache (`45 MB`), cuBLAS workspace, and context |

Comparing `decode_e2e_tok_s_median` (`116.5 tok/s`) against `decode_forward_tok_s` (`125.9 tok/s`) shows that copying `49152 x sizeof(float)` (`192 KB`) of logits to host memory and synchronizing every token costs about `0.6 ms` per step (`~8%` throughput). Moving argmax/top-p sampling to a device kernel would recover most of that gap.

## Build and usage

### Prerequisites

- CMake 3.24+ (configured for 4.4.3 in `CMakeLists.txt`)
- CUDA Toolkit (with `cublas` and `cudart`)
- C++17 compiler (MSVC on Windows or GCC/Clang on Linux)
- Rust toolchain (required by `tokenizers-cpp` to build the Hugging Face tokenizer static library)
- Python 3 with `torch` and `safetensors` (for weight export)

### 1. Export model weights

Place `model.safetensors` and `tokenizer.json` from [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) into `dist/`, then run:

```bash
python scripts/helper.py
```

This writes `dist/model.bin`.

### 2. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_NVTX=ON
cmake --build build --config Release
```

The build compiles with `-lineinfo` and optional NVTXv3 ranges (`include/profiling.h`) around `prefill`, `decode_e2e`, `forward`, `layer`, `attention`, `mlp`, `lm_head`, and `sample`, plus `cudaProfilerStart()` / `cudaProfilerStop()` around measured benchmark iterations.

### 3. Run

Interactive multi-turn ChatML session (supports `/reset` and `/exit`):

```bash
./build/Release/ProjectCuda.exe --temp 0.6 --top-p 0.9 --max-tokens 256
```

Single-prompt CLI mode:

```bash
./build/Release/ProjectCuda.exe --prompt "Explain how a transformer language model generates text, one token at a time."
```

Reproducible benchmark:

```bash
./build/Release/ProjectCuda.exe --bench --bench-tokens 128 --bench-runs 5 --bench-warmup 2
```

## Repository layout

- `include/config.h`: Model hyperparameters and 256-byte binary header definition.
- `include/model.cuh`, `src/model.cu`: Weight loading, arena pointer carving, `RunState` allocation, forward pass, and temperature / top-p sampling.
- `include/kernels.cuh`, `src/kernels.cu`: CUDA kernels (`rmsnorm`, `rope`, `append_kv_cache`, `gqa_attn_scores`, `softmax`, `gqa_attn_out`, `swiglu`, `residual_add`) and `cublasGemmEx` wrappers.
- `include/profiling.h`: RAII `NvtxRange` wrapper for Nsight Systems annotations.
- `src/main.cu`: CLI entrypoint, multi-turn ChatML loop, and `run_bench` harness.
- `scripts/helper.py`: Converts `.safetensors` weights into the flat FP16 `model.bin` format.
