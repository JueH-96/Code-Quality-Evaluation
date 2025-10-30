def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Read N, Q
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Read the pattern P (N rows, each of length N)
    P = input_data[2:2+N]
    
    # Build prefix sums for black squares in the N×N pattern
    # prefix[i][j] = number of black squares in the rectangle [0..i-1]×[0..j-1] (inclusive)
    prefix = [[0]*(N+1) for _ in range(N+1)]
    
    for i in range(N):
        row_str = P[i]
        for j in range(N):
            prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j] + (1 if row_str[j] == 'B' else 0)
    
    # Total number of black squares in one full NxN block
    total_black = prefix[N][N]
    
    # Function to get number of black squares in the NxN pattern from (0,0) to (r,c) inclusive
    # where 0 <= r < N, 0 <= c < N
    def sum_in_pattern(r, c):
        return prefix[r+1][c+1]  # using the prefix sums
    
    # count_black(u,v) = number of black squares in the infinite tiling from (0,0) to (u,v) inclusive
    # If u<0 or v<0, we define it to be 0 (empty rectangle).
    def count_black(u, v):
        if u < 0 or v < 0:
            return 0
        # How many full NxN blocks fit in the row dimension?
        row_block_count = u // N
        # Remainder row index (within 0..N-1)
        row_mod = u % N
        
        # How many full NxN blocks fit in the column dimension?
        col_block_count = v // N
        # Remainder col index (within 0..N-1)
        col_mod = v % N
        
        # 1) Full blocks
        full_blocks_contrib = row_block_count * col_block_count * total_black
        # 2) Partial column strip (rows [0..row_block_count*N - 1], columns [col_block_count*N..v])
        #    => repeated row_block_count times, each strip is from row=0..N-1, col=0..col_mod
        partial_col_contrib = row_block_count * sum_in_pattern(N-1, col_mod)
        # 3) Partial row strip (rows [row_block_count*N..u], columns [0..col_block_count*N - 1])
        #    => repeated col_block_count times, each strip is from row=0..row_mod, col=0..N-1
        partial_row_contrib = col_block_count * sum_in_pattern(row_mod, N-1)
        # 4) Partial corner (rows [row_block_count*N..u], columns [col_block_count*N..v])
        #    => subpattern from row=0..row_mod, col=0..col_mod
        partial_corner_contrib = sum_in_pattern(row_mod, col_mod)
        
        return full_blocks_contrib + partial_col_contrib + partial_row_contrib + partial_corner_contrib
    
    # Process each query
    idx = 2 + N
    out = []
    for _ in range(Q):
        A = int(input_data[idx]);   B = int(input_data[idx+1])
        C = int(input_data[idx+2]); D = int(input_data[idx+3])
        idx += 4
        
        # Using inclusion-exclusion:
        # black_count([A..C]×[B..D]) = count_black(C,D) - count_black(A-1,D)
        #                              - count_black(C,B-1) + count_black(A-1,B-1)
        ans = (
            count_black(C, D)
            - count_black(A - 1, D)
            - count_black(C, B - 1)
            + count_black(A - 1, B - 1)
        )
        out.append(str(ans))
    
    print("
".join(out))

# Do not forget to call main()!
if __name__ == "__main__":
    main()