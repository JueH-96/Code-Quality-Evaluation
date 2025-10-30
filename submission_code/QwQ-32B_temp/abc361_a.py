# Read input values
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert X after the K-th element (1-based index)
b = a[:k] + [x] + a[k:]

# Print the resulting sequence
print(' '.join(map(str, b)))