n, k = map(int, input().split())
a = list(map(int, input().split()))
new_a = a[n - k:] + a[:n - k]
print(' '.join(map(str, new_a)))