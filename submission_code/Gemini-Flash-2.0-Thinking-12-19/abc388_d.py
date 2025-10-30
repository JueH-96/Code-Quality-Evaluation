import sys

# Fenwick Tree (Binary Indexed Tree) implementation
# Supports point updates and prefix sums
# This implementation uses 1-based indexing for the tree structure.
# Application values are mapped directly to tree indices.
# Values X_m = R_m + m + 1 are >= 1.

def update(bit, idx, val):
    """Adds val to the element at value index idx (1-based value)."""
    # tree_idx corresponds to the value itself
    tree_idx = idx
    while tree_idx < len(bit):
        bit[tree_idx] += val
        tree_idx += tree_idx & (-tree_idx)

def query(bit, idx):
    """Gets the prefix sum up to value index idx (1-based value)."""
    # sum up to value idx
    tree_idx = idx
    sum_val = 0
    while tree_idx > 0:
        sum_val += bit[tree_idx]
        tree_idx -= tree_idx & (-tree_idx)
    return sum_val

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Max possible value for X_m = R_m + m + 1
    # R_m <= A_m + m
    # X_m <= A_m + 2m + 1
    # Max A_m = 5e5, Max m = N-1 = 5e5-1
    # Max X_m <= 5e5 + 2*(5e5-1) + 1 = 1_500_000 - 1
    # Need BIT indices covering values from 1 up to approx 1.5e6 - 1.
    # BIT size needs to be max_value + 1 for 1-based value mapping.
    # A safe upper bound like 1.51e6 should be sufficient.
    BIT_MAX_VALUE = 1_500_000 + 10_000 # Safe upper bound for X_m values
    bit = [0] * (BIT_MAX_VALUE + 1) # BIT array is 1-indexed based on value (value v at bit[v])

    R = [0] * N # R[i] will store R_i = A_i + k_{i+1}

    # Calculate R_i for i = 0 to N-1
    # R_i = A_i + k_{i+1}
    # k_{i+1} = count of m in {0, ..., i-1} such that R_m + m + 1 > i
    # k_1 = 0 (for i=0, k_{0+1})

    # For i = 0
    k_next = 0 # k_1
    R[0] = A[0] + k_next
    X_0 = R[0] + 0 + 1
    # X_0 >= 1. Valid value for BIT index.
    # X_0 <= 5e5 + 1, well within BIT_MAX_VALUE.
    update(bit, X_0, 1) # Add X_0 to BIT

    # For i = 1 to N-1
    for i in range(1, N):
        # Calculate k_{i+1} = count of m in {0, ..., i-1} s.t. R_m + m + 1 > i
        # These are the X_m values (for m=0..i-1) already in the BIT.
        # Count X_m > i is (total count of elements processed so far) - (count X_m <= i)
        # Total count of elements processed so far (m=0 to i-1) is i.
        # Count X_m <= i is query(bit, i).
        # Note: X_m are >= 1. query(i) counts values v such that 1 <= v <= i.
        k_next = i - query(bit, i) # k_{i+1}

        R[i] = A[i] + k_next
        X_i = R[i] + i + 1
        # X_i >= 1. Valid value for BIT index.
        # Max X_i <= 1.5e6 - 1. BIT_MAX_VALUE=1.51e6 is safe.
        # If X_i was >= BIT_MAX_VALUE, `update(X_i, 1)` would error or be ignored.
        # If ignored, future `query(j)` where j < X_i would be correct (not counting X_i).
        # If X_i > BIT_MAX_VALUE, it would still contribute to k_next calculation if formula was total-query_le.
        # Using k_next = i - query(i) relies on all X_m for m < i fitting in BIT and query(i) working correctly.
        # Analytical max confirms X_m fits.
        update(bit, X_i, 1) # Add X_i to BIT

    # Calculate final stones B_i for i = 0 to N-1
    # B_i = max(0, R_i - (Number of times alien i is adult and gives stone))
    # Alien i is adult from time i+1 to N. Number of giving opportunities = N - (i+1) = N - i - 1.
    # Final count B_i = max(0, R_i - (N - i - 1))
    
    B = [0] * N
    for i in range(N):
        opportunities = N - i - 1
        B[i] = max(0, R[i] - opportunities)

    # Print the results
    # Output requires space-separated values.
    print(*B)

solve()