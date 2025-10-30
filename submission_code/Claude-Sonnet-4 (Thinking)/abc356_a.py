# YOUR CODE HERE
N, L, R = map(int, input().split())

# Create initial sequence
A = list(range(1, N+1))

# Convert to 0-based indexing for Python
L_index = L - 1
R_index = R - 1

# Reverse the subarray from L_index to R_index (inclusive)
A[L_index:R_index+1] = A[L_index:R_index+1][::-1]

# Print the result
print(' '.join(map(str, A)))