

def get_attn_flops(batch_size, num_heads, seq_len, head_dim) -> int:
    # [b,h,s,d] * [b,h,d,s] -> [b,h,s,s]
    flops = batch_size * num_heads * seq_len * head_dim * seq_len
    # [b,h,s,s] * [b,h,s,d] -> [b,h,s,d]
    flops += batch_size * num_heads * seq_len * seq_len * head_dim
    flops *= 2
    return flops


def main():
    flops = get_attn_flops(2, 24, 4096 + 333, 64)
    print(f"FLOPS: {flops}")

if __name__ == "__main__":
    main()