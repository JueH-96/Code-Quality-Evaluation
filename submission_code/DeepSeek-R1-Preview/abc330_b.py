n, l, r = map(int, input().split())
a = list(map(int, input().split()))
result = [min(r, max(l, x)) for x in a]
print(' '.join(map(str, result)))