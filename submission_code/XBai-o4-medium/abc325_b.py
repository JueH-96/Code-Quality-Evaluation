n = int(input())
bases = [tuple(map(int, input().split())) for _ in range(n)]

max_total = 0

for s in range(24):
    current = 0
    for w, x in bases:
        local_time = (s + x) % 24
        if 9 <= local_time <= 17:
            current += w
    if current > max_total:
        max_total = current

print(max_total)