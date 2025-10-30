# YOUR CODE HERE
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

b = a[:k] + [x] + a[k:]
print(*b)