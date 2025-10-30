import sys

# Function to get digit (0, 1, or 2) from mask at specific position
def get_digit(mask, i, pow3):
    """Extracts the i-th digit (0, 1, or 2) from the base-3 representation of mask."""
    return (mask // pow3[i]) % 3

# Function to check if a column mask is valid based on vertical constraints and fixed cells
def is_column_valid_func(mask, col_data, H, pow3):
    """Checks if a mask is valid for a given column based on vertical adjacency and fixed cells."""
    # Check vertical constraints
    for i in range(1, H):
        if get_digit(mask, i, pow3) == get_digit(mask, i - 1, pow3):
            return False
            
    # Check fixed cells
    for i in range(H):
        char = col_data[i]
        if char != '?':
            # Convert character '1', '2', '3' to digit 0, 1, 2
            required_digit = int(char) - 1
            if get_digit(mask, i, pow3) != required_digit:
                return False
                
    return True

# In-place transform function using the matrix M
def transform_inplace(A, H, pow3, P):
    """Applies the matrix M based transform in-place on vector A."""
    # Transformation matrix M (0 if same, 1 if different) for digits 0, 1, 2
    # M[i][j] = 1 if i != j else 0
    # M = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

    # Apply transform dimension by dimension
    for d in range(H):
        # pow3[d+1] is 3^(d+1), pow3[d] is 3^d
        stride = pow3[d]
        block_size = 3 * stride # Size of a 1D block along dimension d = pow3[d+1]

        # Iterate through all elements, grouped into 3x1 vectors along dimension d
        # The outer loop iterates through blocks of size block_size
        for i in range(0, 3**H, block_size):
             # The inner loop iterates within each block, covering `stride` groups of 3 elements
             for j in range(stride):
                # Get the three indices for the current group along dimension `d`
                # These indices are separated by `stride` = 3^d
                idx0 = i + j
                idx1 = i + j + stride
                idx2 = i + j + 2 * stride

                # Get values
                v0, v1, v2 = A[idx0], A[idx1], A[idx2]

                # Apply matrix multiplication: M * [v0, v1, v2]^T
                # M = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                # new_v0 = 0*v0 + 1*v1 + 1*v2 = v1 + v2
                # new_v1 = 1*v0 + 0*v1 + 1*v2 = v0 + v2
                # new_v2 = 1*v0 + 1*v1 + 0*v2 = v0 + v1
                
                new_v0 = (v1 + v2) % P
                new_v1 = (v0 + v2) % P
                new_v2 = (v0 + v1) % P
                
                # Update values
                A[idx0] = new_v0
                A[idx1] = new_v1
                A[idx2] = new_v2

# Read input
H_orig, W_orig = map(int, sys.stdin.readline().split())
S_orig = [sys.stdin.readline().strip() for _ in range(H_orig)]

H, W = H_orig, W_orig
S = S_orig

# Transpose grid if H > W to make H the smaller dimension (for smaller state space 3^H)
if H > W:
    S_T = [['' for _ in range(H_orig)] for _ in range(W_orig)]
    for i in range(H_orig):
        for j in range(W_orig):
            S_T[j][i] = S_orig[i][j]
    H, W = W_orig, H_orig
    S = S_T

# Modulo constant
P = 998244353

# Precompute powers of 3
max_mask = 3**H
pow3 = [1] * (H + 1)
for i in range(1, H + 1):
    pow3[i] = pow3[i-1] * 3

# DP table: dp[current_col % 2][mask]
# We only need the DP state of the previous column to compute the current column's state
# Using two arrays to save memory: one for the previous state, one for the current state
dp = [[0] * max_mask, [0] * max_mask]

# Compute validity for the first column (j=0)
# This can be done outside the main loop as it's only needed once
is_column_valid_0 = [False] * max_mask
col_data_0 = [S[i][0] for i in range(H)]
for mask in range(max_mask):
    if is_column_valid_func(mask, col_data_0, H, pow3):
        is_column_valid_0[mask] = True

# Initialize DP for the first column (j=0)
curr = 0
for mask in range(max_mask):
    if is_column_valid_0[mask]:
        dp[curr][mask] = 1

# DP loop for j = 1 to W-1
for j in range(1, W):
    prev = curr
    curr = (j % 2)

    # Compute validity for the current column j
    # We need validity for the current column to filter the results after the transform
    is_column_valid_curr = [False] * max_mask
    col_data_j = [S[i][j] for i in range(H)]
    for mask in range(max_mask):
         if is_column_valid_func(mask, col_data_j, H, pow3):
             is_column_valid_curr[mask] = True

    # The vector to be transformed is the DP state of the previous column
    # Apply the transform M on a copy of the previous DP state
    # The transform result corresponds to the sum over compatible previous states
    transformed_dp_prev = list(dp[prev]) # Create a mutable copy for in-place transform
    transform_inplace(transformed_dp_prev, H, pow3, P)

    # Update the current DP state based on the transformed previous state and current column validity
    for mask in range(max_mask):
        if is_column_valid_curr[mask]:
            # transformed_dp_prev[mask] now holds the sum of dp[prev_mask]
            # for all prev_mask such that digit(prev_mask, i) != digit(mask, i) for all i
            dp[curr][mask] = transformed_dp_prev[mask]
        else:
            dp[curr][mask] = 0

# The final answer is the sum of all values in the last column's DP state
total_count = sum(dp[(W - 1) % 2]) % P

# Print the result
print(total_count)