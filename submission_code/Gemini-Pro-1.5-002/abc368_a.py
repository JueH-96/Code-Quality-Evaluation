# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

b = a[n-k:] + a[:n-k]

print(*b)