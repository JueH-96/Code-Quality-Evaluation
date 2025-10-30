n, l, r = map(int, input().split())
a = list(map(int, input().split()))

result = []
for ai in a:
    xi = max(l, min(r, ai))
    result.append(xi)

print(' '.join(map(str, result)))