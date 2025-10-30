# YOUR CODE HERE
# Read inputs
N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Create sequence B by inserting X after the K-th element
B = A[:K] + [X] + A[K:]

# Print the result
print(' '.join(map(str, B)))