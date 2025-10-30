n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

max_streak = 0
current = 0

for day in range(d):
    all_free = True
    for person in range(n):
        if schedules[person][day] != 'o':
            all_free = False
            break
    if all_free:
        current += 1
        if current > max_streak:
            max_streak = current
    else:
        current = 0

print(max_streak)