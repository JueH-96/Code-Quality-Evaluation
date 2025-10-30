# Read input values N, K, X from the first line
N, K, X = map(int, input().split())

# Read the sequence A from the second line
A = list(map(int, input().split()))

# Insert X immediately after the K-th element (1-based index)
# Insert at index K in 0-based indexing
A.insert(K, X)

# Print the modified sequence B with spaces separating the elements
print(' '.join(map(str, A)))