# Read input
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert X after the K-th element (1-based index)
# In Python, slicing is 0-based, so a[:k] gives elements up to index k-1 (K-th element)
# Then insert X, followed by elements from index k onwards
b = a[:k] + [x] + a[k:]

# Output the result
print(' '.join(map(str, b)))