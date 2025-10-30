import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    P_chars = []
    for _ in range(N):
        # Read line, strip newline, convert to list of chars
        P_chars.append(list(sys.stdin.readline().strip()))

    # pattern_prefix_sum[r][c] will store the count of black cells in the pattern P_chars
    # for the rectangle from P_chars[0][0] to P_chars[r-1][c-1].
    # Its dimensions are (N+1)x(N+1) to handle 0-based indexing and boundary conditions easily.
    pattern_prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]

    for r_pat_idx in range(N):  # Iterate 0 to N-1 for pattern rows
        for c_pat_idx in range(N):  # Iterate 0 to N-1 for pattern columns
            is_black = 1 if P_chars[r_pat_idx][c_pat_idx] == 'B' else 0
            
            # pattern_prefix_sum[r][c] refers to subgrid of size r x c.
            # So, P_chars[r_pat_idx][c_pat_idx] contributes to pattern_prefix_sum[r_pat_idx+1][c_pat_idx+1]
            pattern_prefix_sum[r_pat_idx+1][c_pat_idx+1] = is_black + \
                                           pattern_prefix_sum[r_pat_idx][c_pat_idx+1] + \
                                           pattern_prefix_sum[r_pat_idx+1][c_pat_idx] - \
                                           pattern_prefix_sum[r_pat_idx][c_pat_idx]
            
    # This function calculates the number of black squares in the rectangle
    # from (0,0) to (R_abs-1, C_abs-1) in the large grid.
    # R_abs, C_abs are 0-indexed, exclusive upper bounds (i.e., R_abs rows, C_abs columns).
    def count_black_in_prefix_rect(R_abs, C_abs):
        # If the rectangle is empty (0 rows or 0 columns), count is 0.
        if R_abs <= 0 or C_abs <= 0:
            return 0
        
        # Number of full N-row blocks and remaining rows in the R_abs rows
        num_full_row_blocks = R_abs // N
        remaining_rows = R_abs % N
        
        # Number of full N-col blocks and remaining columns in the C_abs columns
        num_full_col_blocks = C_abs // N
        remaining_cols = C_abs % N
        
        # 1. Contribution from full N x N blocks
        # Each such block has pattern_prefix_sum[N][N] black squares.
        # There are num_full_row_blocks * num_full_col_blocks such blocks.
        total_black_count = num_full_row_blocks * num_full_col_blocks * pattern_prefix_sum[N][N]
        
        # 2. Contribution from partial blocks along the right edge:
        # These are num_full_row_blocks blocks, each of size N x remaining_cols.
        # Each such block has pattern_prefix_sum[N][remaining_cols] black squares.
        total_black_count += num_full_row_blocks * pattern_prefix_sum[N][remaining_cols]
        
        # 3. Contribution from partial blocks along the bottom edge:
        # These are num_full_col_blocks blocks, each of size remaining_rows x N.
        # Each such block has pattern_prefix_sum[remaining_rows][N] black squares.
        total_black_count += num_full_col_blocks * pattern_prefix_sum[remaining_rows][N]
        
        # 4. Contribution from the single corner block:
        # This block is of size remaining_rows x remaining_cols.
        # It has pattern_prefix_sum[remaining_rows][remaining_cols] black squares.
        total_black_count += pattern_prefix_sum[remaining_rows][remaining_cols]
        
        return total_black_count

    query_results = []
    for _ in range(Q):
        A, B, C, D = map(int, sys.stdin.readline().split())
        
        # The query is for the rectangle from (A,B) to (C,D) inclusive.
        # Using the principle of inclusion-exclusion with count_black_in_prefix_rect(R,C)
        # which counts black squares in [0,R-1]x[0,C-1]:
        # The sum for cells (r,c) where A <= r <= C and B <= c <= D is:
        #   count_black_in_prefix_rect(C+1, D+1)  (covers cells (r,c) with 0<=r<=C, 0<=c<=D)
        # - count_black_in_prefix_rect(A, D+1)    (subtracts cells with 0<=r<=A-1, 0<=c<=D)
        # - count_black_in_prefix_rect(C+1, B)    (subtracts cells with 0<=r<=C, 0<=c<=B-1)
        # + count_black_in_prefix_rect(A, B)      (adds back cells with 0<=r<=A-1, 0<=c<=B-1,
        #                                          which were subtracted twice)
        
        current_query_ans = count_black_in_prefix_rect(C + 1, D + 1) \
                          - count_black_in_prefix_rect(A, D + 1) \
                          - count_black_in_prefix_rect(C + 1, B) \
                          + count_black_in_prefix_rect(A, B)
        query_results.append(str(current_query_ans))
        
    # Print all results, each on a new line.
    sys.stdout.write("
".join(query_results) + "
")

# Execute the solution
solve()