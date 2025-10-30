n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

all_free = []
for day in range(d):
    free = True
    for person in range(n):
        if schedules[person][day] == 'x':
            free = False
            break
    all_free.append(1 if free else 0)

max_streak = 0
current = 0

for day in all_free:
    if day == 1:
        current += 1
        if current > max_streak:
            max_streak = current
    else:
        current = 0

print(max_streak)