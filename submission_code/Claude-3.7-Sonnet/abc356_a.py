# Read input
N, L, R = map(int, input().split())

# Create initial sequence A = [1, 2, ..., N]
A = list(range(1, N+1))

# Reverse the elements from position L to position R
# Note: Python uses 0-based indexing, so we adjust the positions
A[L-1:R] = reversed(A[L-1:R])

# Print the sequence after the operation
print(*A)