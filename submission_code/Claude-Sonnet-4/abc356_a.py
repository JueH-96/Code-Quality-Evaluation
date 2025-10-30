# YOUR CODE HERE
N, L, R = map(int, input().split())

# Create the initial sequence A = (1, 2, ..., N)
A = list(range(1, N + 1))

# Convert to 0-indexed positions
L_idx = L - 1
R_idx = R - 1

# Reverse the elements from L_idx to R_idx (inclusive)
A[L_idx:R_idx + 1] = A[L_idx:R_idx + 1][::-1]

# Print the result
print(*A)