n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

# Determine if all people are free on each day
all_free = []
for j in range(d):
    all_free_day = True
    for i in range(n):
        if schedules[i][j] != 'o':
            all_free_day = False
            break
    all_free.append(all_free_day)

# Find the maximum consecutive True values
max_consecutive = 0
current = 0

for day in all_free:
    if day:
        current += 1
        max_consecutive = max(max_consecutive, current)
    else:
        current = 0

print(max_consecutive)