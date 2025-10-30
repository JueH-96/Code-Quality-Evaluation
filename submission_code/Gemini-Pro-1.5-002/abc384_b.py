n, r = map(int, input().split())
for _ in range(n):
    d, a = map(int, input().split())
    if d == 1:
        if 1600 <= r <= 2799:
            r += a
    elif d == 2:
        if 1200 <= r <= 2399:
            r += a
print(r)