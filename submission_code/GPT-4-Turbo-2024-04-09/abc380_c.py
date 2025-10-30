def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Find all 1-blocks
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            start = i
            while i < N and S[i] == '1':
                i += 1
            end = i - 1
            blocks.append((start, end))
        else:
            i += 1
    
    # We need to move the K-th block (0-based index K-1)
    kth_block_start, kth_block_end = blocks[K-1]
    kth_block_length = kth_block_end - kth_block_start + 1
    
    if K > 1:
        prev_block_end = blocks[K-2][1]
    else:
        prev_block_end = -1
    
    # Construct the result
    result = []
    if prev_block_end >= 0:
        result.append(S[:prev_block_end + 1])  # Include everything up to the end of the (K-1)-th block
    
    # Insert the K-th block immediately after the (K-1)-th block
    result.append('1' * kth_block_length)
    
    # Add zeros that were between (K-1)-th and K-th block if any
    if K > 1:
        result.append('0' * (kth_block_start - prev_block_end - 1))
    
    # Add the part between the K-th block and the next block
    if kth_block_end + 1 < N:
        result.append(S[kth_block_end + 1:])
    
    # Print the result
    print(''.join(result))

if __name__ == "__main__":
    main()