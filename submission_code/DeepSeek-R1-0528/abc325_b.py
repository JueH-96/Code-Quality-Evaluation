n = int(input().strip())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_participate = 0
for t in range(24):
    total_employees = 0
    for w, x in bases:
        local_time = (t + x) % 24
        if 9 <= local_time <= 17:
            total_employees += w
    if total_employees > max_participate:
        max_participate = total_employees

print(max_participate)