def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Find all 1-blocks
    blocks = []
    current_start = None
    for i in range(1, N + 1):
        if S[i - 1] == '1':
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                blocks.append((current_start, i - 1))
                current_start = None
    if current_start is not None:
        blocks.append((current_start, N))
    
    # Ensure there are at least K blocks
    if len(blocks) < K:
        print(S)
        return
    
    # Get block_{K-1} and block_K
    block_K_minus_1 = blocks[K - 2]
    block_K = blocks[K - 1]
    
    l_K_minus_1, r_K_minus_1 = block_K_minus_1
    l_K, r_K = block_K
    
    # Convert S to list for easy modification
    S_list = list(S)
    
    # Set block_K to '0's
    for i in range(l_K - 1, r_K):
        S_list[i] = '0'
    
    # Calculate the insertion position
    insert_start = r_K_minus_1 - 1  # 0-based index
    insert_length = r_K - l_K + 1
    
    # Set the insertion part to '1's
    for i in range(insert_start + 1, insert_start + 1 + insert_length):
        if i < N:
            S_list[i] = '1'
    
    # If insert_start + insert_length + 1 < r_K, set those positions to '0's
    insert_end = insert_start + 1 + insert_length
    if insert_end < r_K - 1:
        for i in range(insert_end + 1, r_K):
            S_list[i] = '0'
    
    # Print the modified string
    print(''.join(S_list))

if __name__ == "__main__":
    main()