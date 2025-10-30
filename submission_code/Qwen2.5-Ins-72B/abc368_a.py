# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Perform the operation
A = A[-K:] + A[:-K]

# Print the result
print(' '.join(map(str, A)))