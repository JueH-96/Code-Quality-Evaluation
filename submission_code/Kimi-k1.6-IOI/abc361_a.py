# Read input
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert x after the k-th element (1-based index)
b = a[:k] + [x] + a[k:]

# Print the result
print(' '.join(map(str, b)))