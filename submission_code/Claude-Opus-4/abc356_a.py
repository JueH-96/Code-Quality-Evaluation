# YOUR CODE HERE
N, L, R = map(int, input().split())

# Create the initial sequence
A = list(range(1, N + 1))

# Reverse the L-th through R-th elements (convert to 0-based indexing)
# L-1 is the starting index (0-based), R is the ending index (exclusive in Python slicing)
A[L-1:R] = A[L-1:R][::-1]

# Print the result
print(' '.join(map(str, A)))