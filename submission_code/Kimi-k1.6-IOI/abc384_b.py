n, r = map(int, input().split())
contests = [tuple(map(int, input().split())) for _ in range(n)]

current = r
for d, a in contests:
    if d == 1:
        low, high = 1600, 2799
    else:
        low, high = 1200, 2399
    if low <= current <= high:
        current += a
print(current)