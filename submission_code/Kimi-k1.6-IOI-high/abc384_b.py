n, r = map(int, input().split())
contests = [tuple(map(int, input().split())) for _ in range(n)]

current = r
for d, a in contests:
    if d == 1:
        if 1600 <= current <= 2799:
            current += a
    else:
        if 1200 <= current <= 2399:
            current += a
print(current)