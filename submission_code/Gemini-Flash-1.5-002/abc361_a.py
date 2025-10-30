# YOUR CODE HERE
n, k, x = map(int, input().split())
a = list(map(int, input().split()))

b = a[:k]
b.append(x)
b.extend(a[k:])

print(*b)