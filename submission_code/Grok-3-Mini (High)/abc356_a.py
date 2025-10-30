# Read input values
N, L, R = map(int, input().split())

# Create the initial sequence
A = list(range(1, N + 1))

# Reverse the sublist from index L to R (1-based indexing)
# Which corresponds to Python slice A[L-1:R]
A[L-1:R] = A[L-1:R][::-1]

# Print the modified sequence with spaces
print(' '.join(map(str, A)))