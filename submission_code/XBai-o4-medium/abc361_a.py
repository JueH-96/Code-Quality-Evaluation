# Read input values
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert X after the K-th element (1-based)
new_a = a[:k] + [x] + a[k:]

# Print the result
print(' '.join(map(str, new_a)))