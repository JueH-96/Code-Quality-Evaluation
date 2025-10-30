def find_1_blocks(s, n):
    """Find all 1-blocks in the string s."""
    blocks = []
    i = 0
    while i < n:
        if s[i] == '1':
            start = i
            while i < n and s[i] == '1':
                i += 1
            blocks.append((start, i - 1))
        else:
            i += 1
    return blocks

def move_block(s, n, k):
    """Move the K-th 1-block after the (K-1)-th 1-block."""
    blocks = find_1_blocks(s, n)
    
    k_minus_1_block = blocks[k-2]  # 0-indexed
    k_block = blocks[k-1]  # 0-indexed
    
    r_k_minus_1 = k_minus_1_block[1]
    l_k = k_block[0]
    r_k = k_block[1]
    
    # Part up to and including the (K-1)-th 1-block
    part1 = s[:r_k_minus_1 + 1]
    
    # K-th 1-block placed right after the (K-1)-th 1-block
    part2 = '1' * (r_k - l_k + 1)
    
    # Positions from the end of the moved K-th 1-block to the original end of the K-th 1-block filled with 0s
    part3 = '0' * (l_k - r_k_minus_1 - 1)
    
    # Rest of the string after the K-th 1-block
    part4 = s[r_k + 1:]
    
    return part1 + part2 + part3 + part4

def main():
    n, k = map(int, input().split())
    s = input().strip()
    result = move_block(s, n, k)
    print(result)

if __name__ == "__main__":
    main()