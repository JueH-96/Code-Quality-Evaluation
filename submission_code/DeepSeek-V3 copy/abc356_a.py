# Read input values
N, L, R = map(int, input().split())

# Initialize the sequence A
A = list(range(1, N+1))

# Reverse the sublist from L-th to R-th elements
# Note: Python uses 0-based indexing, so we need to adjust L and R
A[L-1:R] = A[L-1:R][::-1]

# Print the result
print(' '.join(map(str, A)))