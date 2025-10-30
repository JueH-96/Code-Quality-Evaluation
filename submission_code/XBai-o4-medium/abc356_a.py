# Read input
N, L, R = map(int, input().split())

# Create the initial sequence
A = list(range(1, N + 1))

# Perform the reversal operation
modified = A[:L-1] + A[L-1:R][::-1] + A[R:]

# Print the result
print(' '.join(map(str, modified)))