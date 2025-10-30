# YOUR CODE HERE
N, L, R = map(int, input().split())
A = list(range(1, N + 1))

# Reverse the elements from index L-1 to R-1
A[L-1:R] = A[L-1:R][::-1]

# Print the resulting sequence
print(*A)