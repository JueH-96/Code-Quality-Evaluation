n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

all_free = []
for day in range(d):
    free = True
    for person in range(n):
        if schedules[person][day] != 'o':
            free = False
            break
    all_free.append(free)

max_consecutive = 0
current = 0

for status in all_free:
    if status:
        current += 1
        if current > max_consecutive:
            max_consecutive = current
    else:
        current = 0

print(max_consecutive)