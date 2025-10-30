# Read input values
N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Insert X after the K-th element
B = A[:K] + [X] + A[K:]

# Print the resulting sequence
print(' '.join(map(str, B)))