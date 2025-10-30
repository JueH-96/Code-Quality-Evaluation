# YOUR CODE HERE
N, Q = map(int, input().split())
P = []
for i in range(N):
    P.append(input().strip())

# Create prefix sum array
prefix = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        prefix[i][j] = prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
        if P[i-1][j-1] == 'B':
            prefix[i][j] += 1

def count_from_origin(r, c):
    if r < 0 or c < 0:
        return 0
    
    # We're counting from (0,0) to (r,c) inclusive
    rows = r + 1
    cols = c + 1
    
    # Number of complete N x N blocks
    full_row_blocks = rows // N
    full_col_blocks = cols // N
    
    # Remaining partial rows and columns
    partial_rows = rows % N
    partial_cols = cols % N
    
    # Count from complete blocks
    blacks_in_pattern = prefix[N][N]
    total = full_row_blocks * full_col_blocks * blacks_in_pattern
    
    # Add partial column strip (full row blocks, partial columns)
    if partial_cols > 0:
        total += full_row_blocks * prefix[N][partial_cols]
    
    # Add partial row strip (partial rows, full column blocks)
    if partial_rows > 0:
        total += full_col_blocks * prefix[partial_rows][N]
    
    # Add corner (partial rows, partial columns)
    if partial_rows > 0 and partial_cols > 0:
        total += prefix[partial_rows][partial_cols]
    
    return total

# Process queries
for _ in range(Q):
    A, B, C, D = map(int, input().split())
    result = count_from_origin(C, D) - count_from_origin(A-1, D) - count_from_origin(C, B-1) + count_from_origin(A-1, B-1)
    print(result)