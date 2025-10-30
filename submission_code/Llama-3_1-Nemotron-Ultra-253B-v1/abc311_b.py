n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

max_days = 0
current = 0

for j in range(d):
    all_free = True
    for i in range(n):
        if schedules[i][j] != 'o':
            all_free = False
            break
    if all_free:
        current += 1
        if current > max_days:
            max_days = current
    else:
        current = 0

print(max_days)