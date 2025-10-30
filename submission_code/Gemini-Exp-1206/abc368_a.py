n, k = map(int, input().split())
a = list(map(int, input().split()))

result = a[n - k:] + a[:n - k]

print(*result)