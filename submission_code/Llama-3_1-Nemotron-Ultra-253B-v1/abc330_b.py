n, l, r = map(int, input().split())
a = list(map(int, input().split()))
x = [min(r, max(l, ai)) for ai in a]
print(' '.join(map(str, x)))