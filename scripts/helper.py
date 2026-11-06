from io import BufferedWriter
import struct
import torch
from safetensors import safe_open

model_path = "C:/Users/somoa/Desktop/Files/Projects/CPP/CUDA/smoll-inference-engine/dist/model.safetensors"
output_bin = "model.bin"

# SmolLM2-135M Architecture Parameters
MAGIC = 0x534D4F4C  # "SMOL" magic identifier
VOCAB_SIZE = 49152
DIM = 576
INTERMEDIATE_DIM = 1536
N_LAYERS = 30
N_HEADS = 9
N_KV_HEADS = 3
HEAD_DIM = 64
MAX_SEQ_LEN = 2048
DTYPE_FLAG = 1

def write_tensor(file_handle: BufferedWriter, tensor: torch.Tensor, to_fp16=True):
    t = tensor.half() if to_fp16 else tensor.float()
    t = t.contiguous()
    file_handle.write(t.numpy().tobytes())

with safe_open(model_path, framework="pt", device="cpu") as f:
    with open(output_bin, "wb") as out:
        # --- 1. Write Header (Fixed 256-byte metadata block) ---
        # 10 ints = 40 bytes. Pad remaining 216 bytes with zeros.
        header_data = struct.pack(
            "10i",
            MAGIC,
            VOCAB_SIZE,
            DIM,
            INTERMEDIATE_DIM,
            N_LAYERS,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            MAX_SEQ_LEN,
            DTYPE_FLAG,
        )
        out.write(header_data)
        out.write(b"\x00" * (256 - len(header_data)))

        # --- 2. Write Global Tensors ---
        print("Writing embedding table...")
        write_tensor(out, f.get_tensor("model.embed_tokens.weight"))

        print("Writing final layernorm...")
        write_tensor(out, f.get_tensor("model.norm.weight"))

        # --- 3. Write Decoder Blocks ---
        for i in range(N_LAYERS):
            print(f"Writing layer {i}/{N_LAYERS}...")
            prefix = f"model.layers.{i}."
            
            # Pre-attention RMSNorm
            write_tensor(out, f.get_tensor(prefix + "input_layernorm.weight"))
            
            # Attention Projections
            write_tensor(out, f.get_tensor(prefix + "self_attn.q_proj.weight"))
            write_tensor(out, f.get_tensor(prefix + "self_attn.k_proj.weight"))
            write_tensor(out, f.get_tensor(prefix + "self_attn.v_proj.weight"))
            write_tensor(out, f.get_tensor(prefix + "self_attn.o_proj.weight"))
            
            # Post-attention RMSNorm
            write_tensor(out, f.get_tensor(prefix + "post_attention_layernorm.weight"))
            
            # MLP Projections (SwiGLU)
            write_tensor(out, f.get_tensor(prefix + "mlp.gate_proj.weight"))
            write_tensor(out, f.get_tensor(prefix + "mlp.up_proj.weight"))
            write_tensor(out, f.get_tensor(prefix + "mlp.down_proj.weight"))

print(f"Export finished successfully: {output_bin}")