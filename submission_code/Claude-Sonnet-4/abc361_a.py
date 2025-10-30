# YOUR CODE HERE
# Read input
N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Insert X after the K-th element
# This means we take first K elements, add X, then add the rest
B = A[:K] + [X] + A[K:]

# Print the result
print(*B)