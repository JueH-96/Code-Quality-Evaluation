import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])

schedules = data[2:]

max_free_days = 0
current_free_days = 0

for day in range(D):
    all_free = True
    for person in range(N):
        if schedules[person][day] == 'x':
            all_free = False
            break
    if all_free:
        current_free_days += 1
    else:
        max_free_days = max(max_free_days, current_free_days)
        current_free_days = 0

max_free_days = max(max_free_days, current_free_days)

print(max_free_days)