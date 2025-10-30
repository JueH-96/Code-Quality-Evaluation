n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

all_free = []
for j in range(d):
    free = True
    for i in range(n):
        if schedules[i][j] != 'o':
            free = False
            break
    all_free.append(free)

max_len = 0
current = 0
for day in all_free:
    if day:
        current += 1
        if current > max_len:
            max_len = current
    else:
        current = 0

print(max_len)