# YOUR CODE HERE

# Read the inputs
N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Insert X after the K-th element
A.insert(K, X)

# Print the sequence B
print(*A)