n, l, r = map(int, input().split())
a = list(range(1, n+1))
start = l - 1
end = r
a[start:end] = a[start:end][::-1]
print(' '.join(map(str, a)))