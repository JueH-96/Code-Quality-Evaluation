n, l, r = map(int, input().split())
a = list(map(int, input().split()))
res = [max(l, min(ai, r)) for ai in a]
print(' '.join(map(str, res)))