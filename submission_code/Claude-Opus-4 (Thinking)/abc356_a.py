# YOUR CODE HERE
# Read input
N, L, R = map(int, input().split())

# Create initial sequence
A = list(range(1, N + 1))

# Reverse elements from L to R (convert to 0-based indexing)
A[L-1:R] = A[L-1:R][::-1]

# Print the result
print(' '.join(map(str, A)))