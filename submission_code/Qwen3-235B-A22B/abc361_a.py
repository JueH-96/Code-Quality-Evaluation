# Read input values
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

# Insert X after the K-th element (0-based index k in Python)
new_list = a[:k] + [x] + a[k:]

# Output the new list
print(' '.join(map(str, new_list)))