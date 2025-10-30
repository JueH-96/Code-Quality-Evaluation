n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

good_days = []
for j in range(d):
    all_free = True
    for i in range(n):
        if schedules[i][j] != 'o':
            all_free = False
            break
    good_days.append(1 if all_free else 0)

max_consecutive = 0
current = 0
for day in good_days:
    if day == 1:
        current += 1
        if current > max_consecutive:
            max_consecutive = current
    else:
        current = 0

print(max_consecutive)