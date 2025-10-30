M, D = map(int, input().split())
y, m, d = map(int, input().split())

if d < D:
    d += 1
elif m < M:
    m += 1
    d = 1
else:
    y += 1
    m = 1
    d = 1

print(y, m, d)