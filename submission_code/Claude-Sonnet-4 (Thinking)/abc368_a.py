# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Take K cards from bottom and place on top
result = a[n-k:] + a[:n-k]

# Print result
print(' '.join(map(str, result)))