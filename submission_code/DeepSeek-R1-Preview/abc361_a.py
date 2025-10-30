# Read the input values
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert X after the K-th element
b = a[:k] + [x] + a[k:]

# Print the result
print(' '.join(map(str, b)))