n, k = map(int, input().split())
a = list(map(int, input().split()))
new_a = a[-k:] + a[:-k]
print(' '.join(map(str, new_a)))