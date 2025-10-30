n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_employees = 0

for utc_start in range(24):
    total_employees = 0
    for w, x in bases:
        local_start = (utc_start + x) % 24
        if 9 <= local_start <= 17:
            total_employees += w
    max_employees = max(max_employees, total_employees)

print(max_employees)