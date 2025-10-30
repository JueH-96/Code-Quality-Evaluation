def get_blocks_for_prefix(i, N):
    blocks = []
    while i > 0:
        block_size = i & -i
        i_block = block_size.bit_length() - 1
        l = i - block_size + 1
        if l < 0:
            continue  # Skip invalid blocks
        j = l // (1 << i_block)
        # Ensure the block is within bounds
        r = l + (1 << i_block) - 1
        if r >= (1 << N):
            continue
        blocks.append((i_block, j))
        i -= block_size
    return blocks

def main():
    import sys
    N, L, R = map(int, sys.stdin.readline().split())
    if L > R:
        print("! 0", flush=True)
        return
    if L == 0:
        prefix_R_blocks = get_blocks_for_prefix(R, N)
        unique_queries = prefix_R_blocks
    else:
        prefix_R_blocks = get_blocks_for_prefix(R, N)
        prefix_L_minus1_blocks = get_blocks_for_prefix(L - 1, N)
        unique_queries = list(set(prefix_R_blocks + prefix_L_minus1_blocks))
    
    queries_dict = {}
    for i, j in unique_queries:
        print(f"? {i} {j}", flush=True)
        T = int(sys.stdin.readline())
        if T == -1:
            return
        queries_dict[(i, j)] = T
    
    sum_R = 0
    for block in prefix_R_blocks:
        sum_R += queries_dict[block]
    
    sum_L_minus1 = 0
    if L > 0:
        for block in prefix_L_minus1_blocks:
            sum_L_minus1 += queries_dict[block]
    
    result = (sum_R - sum_L_minus1) % 100
    print(f"! {result}", flush=True)

if __name__ == "__main__":
    main()