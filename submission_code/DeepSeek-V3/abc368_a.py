# YOUR CODE HERE
# Read input values
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Perform the operation
B = A[-K:] + A[:-K]

# Print the result
print(' '.join(map(str, B)))