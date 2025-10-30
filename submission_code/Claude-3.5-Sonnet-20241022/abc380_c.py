def find_one_blocks(S, N):
    blocks = []
    i = 0
    while i < N:
        # Skip zeros
        while i < N and S[i] == '0':
            i += 1
        if i >= N:
            break
            
        # Found start of a 1-block
        start = i
        while i < N and S[i] == '1':
            i += 1
        # i is now at the end of the 1-block
        blocks.append((start, i-1))
    
    return blocks

def main():
    # Read input
    N, K = map(int, input().split())
    S = input().strip()
    
    # Find all 1-blocks
    blocks = find_one_blocks(S, N)
    
    # Create the result string
    result = ['0'] * N
    
    # First, copy S up to the end of (K-1)th block
    k_minus_1_end = blocks[K-2][1]
    for i in range(k_minus_1_end + 1):
        result[i] = S[i]
    
    # Then, place the K-th block right after (K-1)th block
    k_start, k_end = blocks[K-1]
    block_length = k_end - k_start + 1
    
    # Place the 1s of K-th block
    for i in range(block_length):
        result[k_minus_1_end + 1 + i] = '1'
    
    # Fill zeros in the original position of K-th block
    for i in range(k_start, k_end + 1):
        result[i] = '0'
    
    # Copy the rest of the string
    for i in range(k_end + 1, N):
        result[i] = S[i]
    
    # Print result
    print(''.join(result))

if __name__ == "__main__":
    main()