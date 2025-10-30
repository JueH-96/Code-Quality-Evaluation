import sys

def solve():
    # Read input
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    # Compute inverse permutations (position of value in permutation)
    # P_inv[val] = rank (1-based)
    # If P = [P_1, P_2, ..., P_N], then P_inv[P_k] = k
    # Example: If P = [8, 15, ...], then P_inv[8] = 1, P_inv[15] = 2, ...
    # We use a list of size N + 1 for 1-based indexing convenience
    P_inv = [0] * (N + 1)
    for k in range(N):
        # P[k] is the (k+1)-th element in the input list, corresponding to P_{k+1}
        # The value P[k] is the row index, and its rank is k+1
        P_inv[P[k]] = k + 1

    # Q_inv[val] = rank (1-based)
    # If Q = [Q_1, Q_2, ..., Q_N], then Q_inv[Q_l] = l
    # Example: If Q = [4, 1, ...], then Q_inv[4] = 1, Q_inv[1] = 2, ...
    Q_inv = [0] * (N + 1)
    for k in range(N):
        # Q[k] is the (k+1)-th element in the input list, corresponding to Q_{k+1}
        # The value Q[k] is the column index, and its rank is k+1
        Q_inv[Q[k]] = k + 1

    # Create N x N grid
    # grid[i][j] corresponds to A_{i+1, j+1} (1-based indexing from problem statement)
    grid = [['0'] * N for _ in range(N)]

    # Apply the derived rule: A_{i, j} = 1 if posP[i] + posQ[j] > N, otherwise 0
    # Here, i and j are 1-based row and column indices (from 1 to N)
    # posP[i] is the rank of row i in permutation P (P_inv[i])
    # posQ[j] is the rank of column j in permutation Q (Q_inv[j])
    # In our 0-indexed grid, grid[row_idx][col_idx] corresponds to A_{row_idx+1, col_idx+1}
    # So, we set grid[i][j] (0-indexed indices) based on A_{i+1, j+1}, which depends on posP[i+1] and posQ[j+1]
    for i in range(N): # 0-indexed row index (0 to N-1)
        row_val = i + 1 # 1-based row value (1 to N) corresponding to the row index A_{row_val, ...}
        for j in range(N): # 0-indexed col index (0 to N-1)
            col_val = j + 1 # 1-based col value (1 to N) corresponding to the column index A_{..., col_val}
            
            # Check the condition based on 1-based values (row_val, col_val) and their ranks
            if P_inv[row_val] + Q_inv[col_val] > N:
                grid[i][j] = '1'
            else:
                grid[i][j] = '0'

    # Print the grid row by row
    for row in grid:
        print("".join(row))

# Execute the solve function
solve()