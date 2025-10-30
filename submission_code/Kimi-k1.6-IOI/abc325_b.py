n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_employees = 0

for s in range(24):
    total = 0
    for w, x in bases:
        local_time = (x + s) % 24
        if 9 <= local_time <= 17:
            total += w
    if total > max_employees:
        max_employees = total

print(max_employees)