def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    P = []
    for _ in range(N):
        P.append(data[index])
        index += 1
    
    queries = []
    for _ in range(Q):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        D = int(data[index])
        index += 1
        queries.append((A, B, C, D))
    
    # Precompute the number of black cells in each row of the pattern P
    black_count = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            black_count[i][j] = (black_count[i][j-1] if j > 0 else 0) + (1 if P[i][j] == 'B' else 0)
    
    # Precompute the number of black cells in each (i, j) modulo N block
    # black_in_block[i][j] will store the number of black cells in the block that starts at (i, j)
    black_in_block = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            black_in_block[i][j] = black_count[i][j] + (black_in_block[i-1][j] if i > 0 else 0)
    
    results = []
    for A, B, C, D in queries:
        # Calculate the number of full blocks and partial blocks in the range
        row_start = A % N
        row_end = C % N
        col_start = B % N
        col_end = D % N
        
        full_rows = (C // N) - (A // N) + 1
        full_cols = (D // N) - (B // N) + 1
        
        # Count black squares by considering full blocks and partial blocks
        black_squares = 0
        
        # Full grid contribution
        total_full_blocks = full_rows * full_cols
        total_black_in_full_pattern = black_in_block[N-1][N-1]
        black_squares += total_full_blocks * total_black_in_full_pattern
        
        # Adjust for the top and bottom partial rows if they exist
        if row_start != 0:
            black_squares -= (black_in_block[row_start-1][N-1] * full_cols)
        if row_end != N-1:
            black_squares -= (black_in_block[N-1][N-1] - black_in_block[row_end][N-1]) * full_cols
        
        # Adjust for the left and right partial columns if they exist
        if col_start != 0:
            black_squares -= (black_in_block[N-1][col_start-1] * full_rows)
        if col_end != N-1:
            black_squares -= (black_in_block[N-1][N-1] - black_in_block[N-1][col_end]) * full_rows
        
        # Add back the intersection of top-left, top-right, bottom-left, bottom-right if double subtracted
        if row_start != 0 and col_start != 0:
            black_squares += black_in_block[row_start-1][col_start-1]
        if row_start != 0 and col_end != N-1:
            black_squares += black_in_block[row_start-1][N-1] - black_in_block[row_start-1][col_end]
        if row_end != N-1 and col_start != 0:
            black_squares += black_in_block[N-1][col_start-1] - black_in_block[row_end][col_start-1]
        if row_end != N-1 and col_end != N-1:
            black_squares += black_in_block[row_end][col_end]
        
        results.append(black_squares)
    
    # Output results
    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()