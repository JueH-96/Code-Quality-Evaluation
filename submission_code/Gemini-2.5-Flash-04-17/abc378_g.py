import sys

def inv(a, m):
    """Modular inverse using Fermat's Little Theorem."""
    return pow(a, m - 2, m)

def factorial(n, m):
    """Calculates factorial n! modulo m."""
    res = 1
    for i in range(1, n + 1):
        res = (res * i) % m
    return res

def prod_hooks(shape, m):
    """Calculates the product of hook lengths for a given Young diagram shape modulo m."""
    prod = 1
    rows = len(shape)
    if rows == 0:
        return 1 # Empty shape has product of hooks 1

    # Find the maximum column index to build column heights
    cols = 0
    for row in shape:
        if row > cols:
            cols = row
            
    if cols == 0:
        return 1 # Shape contains only empty rows, effectively empty

    # Calculate column heights (number of cells in each column)
    col_heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if c < shape[r]:
                col_heights[c] += 1
    
    # Calculate product of hook lengths
    for r in range(rows): # 0-indexed row
        for c in range(shape[r]): # 0-indexed column
            # Hook length for cell (r, c) is
            # (number of cells to the right in the same row)
            # + (number of cells below in the same column)
            # + 1 (for the cell itself)
            right_arm = shape[r] - 1 - c
            down_arm = col_heights[c] - 1 - r
            hook = right_arm + down_arm + 1
            
            prod = (prod * hook) % m

    return prod

def calculate_f_lambda(shape, N, M):
    """Calculates the number of Standard Young Tableaux for a shape with N cells modulo M."""
    if N == 0: # Number of SYT for empty shape is 1
        return 1
        
    fact_N = factorial(N, M)
    hooks_prod = prod_hooks(shape, M)
    
    # If hooks_prod is 0, it means one of the hook lengths is a multiple of M.
    # For prime M and hooks < M, hooks_prod will not be 0.
    # Problem constraints guarantee M is prime and AB <= 120,
    # so hook lengths (<= A+B-1 <= 120+120-1 = 239) are less than M (>= 10^8).
    # So hooks_prod will be non-zero modulo M.
    
    return (fact_N * inv(hooks_prod, M)) % M

# Read input
A, B, M = map(int, sys.stdin.readline().split())

# Total number of cells in the permutation domain {1, ..., AB-1}
N = A * B - 1

# The shape of the RSK insertion tableau P for permutations of {1, ..., N}
# with LIS length A and LDS length B must be lambda, a partition of N,
# with lambda_1 = A and lambda'_1 = B.
# For N = AB-1, the only such shape is the A x B rectangle minus cell (A, B).
# Represented as row lengths: B rows, first B-1 rows of length A, last row of length A-1.
# Note: RSK uses shape (A^B), i.e., A rows of length B. The shape here is (B x A) related.
# LIS length A and LDS length B for a permutation of length N = AB-1 means
# the shape is a partition of N with largest part A and B parts.
# The shape is lambda = (A, A, ..., A, A-1), with B rows.
# This means lambda_1 = A, lambda_B = A-1.
# The number of rows is B, so lambda'_1 = B.
# Shape lambda represented as a list of row lengths:
shape_lambda = [A] * (B - 1) + [A - 1]

# The number of permutations satisfying conditions 1 and 2 is (f^lambda)^2.
# Condition 3 requires that there exists an integer n such that appending n+0.5
# does not change LIS and LDS lengths. This translates to a condition on the
# values in the SYT insertion tableau P: v_(B, A-1) < v_(1, A).
# The number of SYT P of shape lambda satisfying v_(B, A-1) < v_(1, A) is equal
# to the number of SYT of shape mu, where mu is lambda with cell (B, A-1) removed? No.
# The number of SYT P of shape lambda satisfying v_(B, A-1) < v_(1, A) is equal
# to the number of SYT of shape (B x A) \ {(B,A), (B,A-1)}.
# This shape mu is a Young diagram with row lengths:
# A for rows 1 to B-1, and A-2 for row B.
# Represented as row lengths: [A, A, ..., A, A-2], with B elements.
shape_mu = [A] * (B - 1) + [A - 2]

# Ensure shape_mu is a valid Young diagram (row lengths are non-increasing)
# and handle cases where A-2 < 0.
# If A=2, A-2=0. The last row has length 0, effectively empty.
# If A=1, A-2=-1. This shape is invalid, but constraints say A >= 2.
# If A >= 2, then A-2 >= 0.
# The number of cells in mu is (B-1)*A + (A-2) = AB - A + A - 2 = AB-2 = N-1.

# Calculate f^lambda and f^mu modulo M
f_lambda = calculate_f_lambda(shape_lambda, N, M)
f_mu = calculate_f_lambda(shape_mu, N - 1, M) # Mu has N-1 cells

# The total count is f^lambda * (Number of SYT P satisfying condition 3)
# The number of SYT P satisfying condition 3 is f^mu.
result = (f_lambda * f_mu) % M

# Print the result
print(result)