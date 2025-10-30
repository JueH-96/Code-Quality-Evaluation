# Read the input
N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Insert X after the K-th element
B = A[:K] + [A[K-1], X] + A[K:]

# Print the output
print(" ".join(map(str, B)))