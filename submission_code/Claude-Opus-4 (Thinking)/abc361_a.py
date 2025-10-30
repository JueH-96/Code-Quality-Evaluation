# YOUR CODE HERE
# Read N, K, X
N, K, X = map(int, input().split())

# Read the sequence A
A = list(map(int, input().split()))

# Insert X after the K-th element
# First K elements + X + remaining elements
B = A[:K] + [X] + A[K:]

# Print the result
print(' '.join(map(str, B)))