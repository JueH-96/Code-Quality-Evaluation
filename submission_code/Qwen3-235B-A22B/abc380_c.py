def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    S = input[2]
    
    blocks = []
    for i in range(N):
        if S[i] == '1':
            # Check if start of block
            if i == 0 or S[i-1] != '1':
                start = i + 1  # 1-based
            # Check if end of block
            if i == N-1 or S[i+1] != '1':
                end = i + 1  # 1-based
                blocks.append((start, end))
    
    prev_block = blocks[K-2]
    current_block = blocks[K-1]
    prev_end = prev_block[1]
    current_start, current_end = current_block
    
    L = current_end - current_start + 1
    Z = current_start - prev_end - 1
    
    prefix = S[:prev_end]
    inserted = '1' * L + '0' * Z
    suffix = S[current_end:]
    
    print(prefix + inserted + suffix)

if __name__ == "__main__":
    main()