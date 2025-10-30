n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_total = 0

for s in range(24):
    total = 0
    for w, x in bases:
        local_start = (x + s) % 24
        if 9 <= local_start <= 17:
            total += w
    if total > max_total:
        max_total = total

print(max_total)