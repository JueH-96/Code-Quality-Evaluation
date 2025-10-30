n, r = map(int, input().split())
current = r
for _ in range(n):
    d, a = map(int, input().split())
    if d == 1:
        if 1600 <= current <= 2799:
            current += a
    else:
        if 1200 <= current <= 2399:
            current += a
print(current)