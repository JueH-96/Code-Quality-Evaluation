N, K, X = map(int, input().split())
A = list(map(int, input().split()))

# Insert X immediately after the K-th element
A.insert(K, X)

# Print the resulting sequence
print(' '.join(map(str, A)))