n = int(input())
bases = [tuple(map(int, input().split())) for _ in range(n)]

max_employees = 0
for t in range(24):
    current = 0
    for w, x in bases:
        local = (t + x) % 24
        if 9 <= local <= 17:
            current += w
    if current > max_employees:
        max_employees = current

print(max_employees)