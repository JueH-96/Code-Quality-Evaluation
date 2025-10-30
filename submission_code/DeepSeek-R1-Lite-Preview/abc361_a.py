# Read N, K, X from the first line
N, K, X = map(int, input().split())

# Read the sequence A from the second line
A = list(map(int, input().split()))

# Insert X after the K-th element (index K in the list)
A.insert(K, X)

# Print the new sequence B
print(' '.join(map(str, A)))