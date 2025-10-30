n, k = map(int, input().split())
a = list(map(int, input().split()))

bottom_k = a[n-k:]
remaining = a[:n-k]

result = bottom_k + remaining

print(*result)