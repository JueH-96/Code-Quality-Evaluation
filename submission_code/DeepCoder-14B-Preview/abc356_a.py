n, l, r = map(int, input().split())
a = list(range(1, n + 1))
b = a.copy()
b[l-1:r] = b[l-1:r][::-1]
print(' '.join(map(str, b)))