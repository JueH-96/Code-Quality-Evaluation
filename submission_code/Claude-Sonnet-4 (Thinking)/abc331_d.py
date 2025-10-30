N, Q = map(int, input().split())
P = []
for i in range(N):
    P.append(input().strip())

# Precompute prefix sums for the pattern
prefix = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(N):
    for j in range(N):
        prefix[i + 1][j + 1] = prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j]
        if P[i][j] == 'B':
            prefix[i + 1][j + 1] += 1

def count_in_pattern(r1, c1, r2, c2):
    if r1 > r2 or c1 > c2:
        return 0
    return prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1] - prefix[r2 + 1][c1] + prefix[r1][c1]

total_black = prefix[N][N]

def prefix_count(r, c):
    if r < 0 or c < 0:
        return 0
    
    rows = r + 1
    cols = c + 1
    
    complete_row_blocks = rows // N
    complete_col_blocks = cols // N
    remaining_rows = rows % N
    remaining_cols = cols % N
    
    result = 0
    
    # Complete blocks
    result += complete_row_blocks * complete_col_blocks * total_black
    
    # Right edge (complete row blocks, partial columns)
    if remaining_cols > 0:
        result += complete_row_blocks * count_in_pattern(0, 0, N - 1, remaining_cols - 1)
    
    # Bottom edge (partial rows, complete column blocks)
    if remaining_rows > 0:
        result += complete_col_blocks * count_in_pattern(0, 0, remaining_rows - 1, N - 1)
    
    # Bottom-right corner (partial rows and columns)
    if remaining_rows > 0 and remaining_cols > 0:
        result += count_in_pattern(0, 0, remaining_rows - 1, remaining_cols - 1)
    
    return result

def count_black(r1, c1, r2, c2):
    return (prefix_count(r2, c2) - prefix_count(r1 - 1, c2) - 
            prefix_count(r2, c1 - 1) + prefix_count(r1 - 1, c1 - 1))

for _ in range(Q):
    A, B, C, D = map(int, input().split())
    print(count_black(A, B, C, D))