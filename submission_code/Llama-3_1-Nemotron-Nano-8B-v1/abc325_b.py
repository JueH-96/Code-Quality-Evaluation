n = int(input())
employees = [tuple(map(int, input().split())) for _ in range(n)]

max_total = 0
for s in range(9, 18):
    total = 0
    for w, x in employees:
        local_start = (x + s) % 24
        if 9 <= local_start <= 17:
            total += w
    max_total = max(max_total, total)

print(max_total)