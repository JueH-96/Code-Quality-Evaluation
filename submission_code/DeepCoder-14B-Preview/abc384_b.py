n, r = map(int, input().split())
contests = [tuple(map(int, input().split())) for _ in range(n)]

t = r
for d, a in contests:
    if d == 1:
        if 1600 <= t <= 2799:
            t += a
    else:
        if 1200 <= t <= 2399:
            t += a

print(t)