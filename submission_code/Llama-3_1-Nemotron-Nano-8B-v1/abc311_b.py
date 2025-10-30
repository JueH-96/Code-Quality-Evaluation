n, d = map(int, input().split())
people = [input().strip() for _ in range(n)]

max_consec = 0
current = 0
free_days = []

for day in range(1, d + 1):
    all_free = True
    for s in people:
        if s[day - 1] != 'o':
            all_free = False
            break
    if all_free:
        free_days.append(True)
    else:
        free_days.append(False)

current = 0
max_consec = 0
for is_free in free_days:
    if is_free:
        current += 1
        max_consec = max(max_consec, current)
    else:
        current = 0

print(max_consec)