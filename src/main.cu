#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>
#include <tokenizers_cpp.h>
#include <config.h>
#include <model.cuh>
#include <utils.cuh>
#include <profiling.h>

using tokenizers::Tokenizer;

constexpr int TOKEN_ENDOFTEXT = 0; // <|endoftext|>
constexpr int TOKEN_IM_START  = 1; // <|im_start|>
constexpr int TOKEN_IM_END    = 2; // <|im_end|>

std::string load_bytes_from_file(const std::string &path)
{
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail())
    {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

void run_turn(SmolLM2Model &model,
              Tokenizer *tok,
              const std::string &turn_prompt,
              int &pos,
              float temperature,
              float top_p,
              int max_new_tokens)
{
    const int max_seq_len = model.get_config().max_seq_len;
    std::vector<int> prompt_ids = tok->Encode(turn_prompt);

    if (pos + static_cast<int>(prompt_ids.size()) >= max_seq_len)
    {
        std::cout << "\n[Context window full (" << max_seq_len << " tokens). Resetting conversation.]\n";
        pos = 0;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Prefill prompt tokens (only compute logits on the final prompt token)
    for (size_t i = 0; i < prompt_ids.size(); ++i)
    {
        bool is_last = (i + 1 == prompt_ids.size());
        model.forward(prompt_ids[i], pos++, is_last);
    }

    // 2. Autoregressive generation loop
    std::vector<int> generated_ids;
    size_t printed_bytes = 0;
    int gen_count = 0;

    std::cout << "\nAssistant: " << std::flush;
    while (gen_count < max_new_tokens && pos < max_seq_len)
    {
        int next_token = model.sample(temperature, top_p);
        if (next_token == TOKEN_IM_END || next_token == TOKEN_ENDOFTEXT)
        {
            // Feed the closing <|im_end|> + newline into KV cache on next turn
            break;
        }

        generated_ids.push_back(next_token);
        std::string decoded = tok->Decode(generated_ids);
        if (decoded.size() > printed_bytes)
        {
            std::cout << decoded.substr(printed_bytes) << std::flush;
            printed_bytes = decoded.size();
        }

        model.forward(next_token, pos++, true);
        ++gen_count;
    }

    // Close the assistant turn in the KV cache for multi-turn continuity
    std::vector<int> suffix_ids = tok->Encode("<|im_end|>\n");
    for (int id : suffix_ids)
    {
        if (pos < max_seq_len)
            model.forward(id, pos++, false);
    }

    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tok_per_sec = (elapsed_sec > 0.0) ? (gen_count / elapsed_sec) : 0.0;

    std::cout << "\n[" << gen_count << " tokens, " << tok_per_sec << " tok/s, ctx=" << pos << "/" << max_seq_len << "]\n";
}

static double percentile(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return v[idx];
}

// Reproducible benchmark: fixed prompt, greedy decoding, EOS ignored so every run generates the same count.
// Reports prefill and decode separately. Decode is measured two ways:
//   - end-to-end per token (forward + logits copy + CPU argmax), what the chat loop actually pays
//   - forward only (no host sync between tokens), closer to the GPU-bound limit
void run_bench(SmolLM2Model &model, Tokenizer *tok, int gen_tokens, int runs, int warmup)
{
    const std::string prompt =
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "<|im_start|>user\nExplain how a transformer language model generates text, one token at a time.<|im_end|>\n"
        "<|im_start|>assistant\n";
    std::vector<int> prompt_ids = tok->Encode(prompt);
    const int max_seq_len = model.get_config().max_seq_len;
    const int n_prompt = static_cast<int>(prompt_ids.size());
    if (n_prompt + gen_tokens >= max_seq_len)
    {
        std::cerr << "prompt + gen tokens exceed context window\n";
        return;
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::vector<double> prefill_ms, decode_tps, fwd_tps, token_ms;

    for (int r = -warmup; r < runs; ++r)
    {
        const bool measured = r >= 0;
        if (r == 0)
            cudaProfilerStart(); // nsys --capture-range=cudaProfilerApi records only measured runs

        NVTX_RANGE(measured ? "bench_run" : "bench_warmup");

        // Prefill (sequential: the engine has no batched prefill yet)
        int pos = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            NVTX_RANGE("prefill");
            for (int i = 0; i < n_prompt; ++i)
                model.forward(prompt_ids[i], pos++, i + 1 == n_prompt);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // End-to-end decode: forward + sample (host sync) per token
        std::vector<double> run_token_ms;
        {
            NVTX_RANGE("decode_e2e");
            for (int i = 0; i < gen_tokens; ++i)
            {
                auto a = std::chrono::high_resolution_clock::now();
                int next = model.sample(0.0f, 1.0f);
                model.forward(next, pos++, true);
                cudaDeviceSynchronize();
                auto b = std::chrono::high_resolution_clock::now();
                run_token_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            }
        }
        double decode_total_ms = std::accumulate(run_token_ms.begin(), run_token_ms.end(), 0.0);

        // Forward only: same work, no per-token host round trip
        pos = n_prompt;
        auto f0 = std::chrono::high_resolution_clock::now();
        {
            NVTX_RANGE("decode_forward_only");
            for (int i = 0; i < gen_tokens; ++i)
                model.forward(prompt_ids.back(), pos++, true);
            cudaDeviceSynchronize();
        }
        auto f1 = std::chrono::high_resolution_clock::now();

        if (measured)
        {
            prefill_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            decode_tps.push_back(gen_tokens / (decode_total_ms / 1000.0));
            fwd_tps.push_back(gen_tokens / std::chrono::duration<double>(f1 - f0).count());
            token_ms.insert(token_ms.end(), run_token_ms.begin(), run_token_ms.end());
        }
    }
    cudaProfilerStop();

    auto median = [](std::vector<double> v) { return percentile(v, 0.5); };
    std::cout << "\n=== Benchmark (prompt=" << n_prompt << " tokens, generated=" << gen_tokens
              << ", runs=" << runs << ", warmup=" << warmup << ") ===\n";
    std::cout << "prefill_ms_median:        " << median(prefill_ms) << "  ("
              << n_prompt / (median(prefill_ms) / 1000.0) << " tok/s, sequential)\n";
    std::cout << "decode_e2e_tok_s_median:  " << median(decode_tps) << "\n";
    std::cout << "token_ms_p50 / p95:       " << percentile(token_ms, 0.5) << " / " << percentile(token_ms, 0.95) << "\n";
    std::cout << "decode_forward_tok_s:     " << median(fwd_tps) << "  (no per-token host sync)\n";
    std::cout << "gpu_mem_in_use_mb:        " << (total_mem - free_mem) / (1024 * 1024) << "\n";
}

int main(int argc, char **argv)
{
    if (checkGPUStatus() != 0)
        return 1;

    std::string tokenizer_path = std::string(ROOT_PATH) + "/dist/tokenizer.json";
    std::string model_path = std::string(ROOT_PATH) + "/dist/model.bin";

    std::cout << "Loading tokenizer from " << tokenizer_path << "...\n";
    auto blob = load_bytes_from_file(tokenizer_path);
    auto tok = Tokenizer::FromBlobJSON(blob);

    std::cout << "Loading model weights from " << model_path << "...\n";
    SmolLM2Config config;
    SmolLM2Model model(config);
    model.load_model_weights(model_path);

    float temperature = 0.0f; // Default greedy decoding (use --temp 0.6 for sampling)
    float top_p = 0.9f;
    int max_new_tokens = 256;
    std::string single_prompt;
    bool bench = false;
    int bench_tokens = 128, bench_runs = 5, bench_warmup = 2;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--bench")
            bench = true;
        else if (arg == "--bench-tokens" && i + 1 < argc)
            bench_tokens = std::stoi(argv[++i]);
        else if (arg == "--bench-runs" && i + 1 < argc)
            bench_runs = std::stoi(argv[++i]);
        else if (arg == "--bench-warmup" && i + 1 < argc)
            bench_warmup = std::stoi(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc)
            single_prompt = argv[++i];
        else if (arg == "--temp" && i + 1 < argc)
            temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc)
            top_p = std::stof(argv[++i]);
        else if (arg == "--max-tokens" && i + 1 < argc)
            max_new_tokens = std::stoi(argv[++i]);
    }

    const std::string system_prompt =
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

    int pos = 0;

    if (bench)
    {
        run_bench(model, tok.get(), bench_tokens, bench_runs, bench_warmup);
        return 0;
    }

    // One-shot CLI mode: ProjectCuda.exe --prompt "What is the capital of Canada?"
    if (!single_prompt.empty())
    {
        std::string full_prompt =
            system_prompt +
            "<|im_start|>user\n" + single_prompt + "<|im_end|>\n" +
            "<|im_start|>assistant\n";
        run_turn(model, tok.get(), full_prompt, pos, temperature, top_p, max_new_tokens);
        return 0;
    }

    // Interactive multi-turn chat loop
    std::cout << "\n======================================================\n";
    std::cout << " SmolLM2-135M CUDA Chat (commands: /reset, /exit)\n";
    std::cout << "======================================================\n";

    while (true)
    {
        std::cout << "\nUser: ";
        std::string user_input;
        if (!std::getline(std::cin, user_input))
            break;

        if (user_input.empty())
            continue;
        if (user_input == "/exit" || user_input == "exit" || user_input == "quit")
            break;
        if (user_input == "/reset")
        {
            pos = 0;
            std::cout << "[Conversation reset.]\n";
            continue;
        }

        std::string turn_prompt;
        if (pos == 0)
            turn_prompt = system_prompt;
        turn_prompt += "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";

        run_turn(model, tok.get(), turn_prompt, pos, temperature, top_p, max_new_tokens);
    }

    return 0;
}