n, k = map(int, input().split())
a = list(map(int, input().split()))
result = a[-k:] + a[:-k]
print(' '.join(map(str, result)))