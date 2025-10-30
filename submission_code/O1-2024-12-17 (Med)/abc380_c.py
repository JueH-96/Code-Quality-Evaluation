def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = data[2]
    
    # Find all 1-blocks in 1-based indexing
    blocks = []
    in_block = False
    start = 0
    for i, ch in enumerate(S, start=1):
        if ch == '1':
            if not in_block:
                in_block = True
                start = i
        else:
            if in_block:
                blocks.append((start, i - 1))
                in_block = False
    if in_block:
        blocks.append((start, N))
    
    # The (K-1)-th block (in 1-based sense) is blocks[K-2]
    # The K-th block is blocks[K-1]
    _, rKminusOne = blocks[K-2]
    lK, rK = blocks[K-1]
    lengthK = rK - lK + 1  # number of '1's in the K-th block
    
    # We will build T in 1-based fashion for clarity
    T = list(' ' + S)  # T[1]..T[N] corresponds to S[0]..S[N-1]
    
    # Overwrite according to the problem's formula:
    # 1) T_i = 1 for i in [r_{K-1}+1 .. r_{K-1} + lengthK]
    start_ones = rKminusOne + 1
    end_ones = rKminusOne + lengthK
    if start_ones <= N:
        for i in range(start_ones, min(end_ones, N) + 1):
            T[i] = '1'
    
    # 2) T_i = 0 for i in [r_{K-1} + lengthK + 1 .. rK]
    start_zeros = end_ones + 1
    end_zeros = rK
    if start_zeros <= end_zeros and start_zeros <= N:
        for i in range(start_zeros, min(end_zeros, N) + 1):
            T[i] = '0'
    
    # Everything else stays as in S
    # Print the result (skip T[0])
    sys.stdout.write(''.join(T[1:]) + '
')


# Don't forget to call main()
if __name__ == "__main__":
    main()