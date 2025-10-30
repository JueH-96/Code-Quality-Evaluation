N, L, R = map(int, input().split())

# Create initial sequence
A = list(range(1, N+1))

# Reverse the subarray from L to R
A[L-1:R] = A[L-1:R][::-1]

# Print result
print(*A)