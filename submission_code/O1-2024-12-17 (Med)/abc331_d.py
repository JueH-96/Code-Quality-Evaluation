def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Read N and Q
    N, Q = map(int, input_data[:2])
    
    # Read the pattern P
    idx = 2
    P = input_data[idx:idx+N]
    idx += N
    
    # Convert P to a 2D array of 0/1 (white/black)
    pattern = [[1 if c == 'B' else 0 for c in row] for row in P]
    
    # Build prefix sums for the pattern
    # prefix[r+1][c+1] will store number of black squares in rectangle (0,0) to (r,c) inclusive
    prefix = [[0]*(N+1) for _ in range(N+1)]
    for r in range(N):
        row_sum = 0
        for c in range(N):
            row_sum += pattern[r][c]
            prefix[r+1][c+1] = prefix[r][c+1] + row_sum
    
    # Function to safely get black count in sub-rectangle within the pattern
    # (r0,c0) .. (r1,c1) inclusive, clipped to [0,N-1]
    def black_sub(r0, c0, r1, c1):
        # If invalid range, return 0
        if r0 > r1 or c0 > c1 or r0 < 0 or c0 < 0:
            return 0
        # Clip to valid region inside pattern
        if r1 >= N: r1 = N-1
        if c1 >= N: c1 = N-1
        if r0 < 0: r0 = 0
        if c0 < 0: c0 = 0
        if r0 > r1 or c0 > c1:
            return 0
        # Use prefix sums
        return prefix[r1+1][c1+1] - prefix[r1+1][c0] - prefix[r0][c1+1] + prefix[r0][c0]

    # Total black squares in one full N x N block
    total_black = black_sub(0, 0, N-1, N-1)
    
    # f(x, y) returns number of black squares in rectangle (0,0) to (x,y) inclusive in the infinite grid
    def F(x, y):
        if x < 0 or y < 0:
            return 0
        # Calculate how many full blocks
        row_blocks = x // N
        col_blocks = y // N
        row_rem = x % N
        col_rem = y % N
        
        # 1) Fully included blocks
        ans = row_blocks * col_blocks * total_black
        
        # 2) Partial columns region (across all full row blocks)
        #    For each of the row_blocks, we add the sub-rectangle of the pattern from rows=0..N-1, cols=0..col_rem
        #    repeated row_blocks times
        ans += row_blocks * black_sub(0, 0, N-1, col_rem)
        
        # 3) Partial rows region (across all full column blocks)
        #    For each of the col_blocks, we add the sub-rectangle of the pattern from rows=0..row_rem, cols=0..N-1
        ans += col_blocks * black_sub(0, 0, row_rem, N-1)
        
        # 4) The corner region (row_rem+1 by col_rem+1)
        ans += black_sub(0, 0, row_rem, col_rem)
        
        return ans
    
    # Process queries
    out = []
    base = idx
    for _ in range(Q):
        A, B, C, D = map(int, input_data[base:base+4])
        base += 4
        # Number of black squares in (A,B) to (C,D)
        # = F(C,D) - F(A-1,D) - F(C,B-1) + F(A-1,B-1)
        ans = F(C, D) - F(A-1, D) - F(C, B-1) + F(A-1, B-1)
        out.append(str(ans))
    
    print("
".join(out))

# Do not forget to call main!
if __name__ == "__main__":
    main()