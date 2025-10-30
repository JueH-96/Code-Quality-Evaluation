n, d = map(int, input().split())
schedules = [input().strip() for _ in range(n)]

all_free = []
for day in range(d):
    all_ok = True
    for s in schedules:
        if s[day] != 'o':
            all_ok = False
            break
    all_free.append(all_ok)

max_count = 0
current = 0
for flag in all_free:
    if flag:
        current += 1
        if current > max_count:
            max_count = current
    else:
        current = 0

print(max_count)