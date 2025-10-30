# Read the input
N, L, R = map(int, input().split())

# Create the initial sequence
A = list(range(1, N+1))

# Reverse the L-th through R-th elements
A[L-1:R] = reversed(A[L-1:R])

# Print the sequence
print(" ".join(map(str, A)))