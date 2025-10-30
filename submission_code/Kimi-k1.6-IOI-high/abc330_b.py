n, l, r = map(int, input().split())
a = list(map(int, input().split()))
result = [max(l, min(x, r)) for x in a]
print(' '.join(map(str, result)))