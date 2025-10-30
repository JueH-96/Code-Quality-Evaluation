# YOUR CODE HERE
N, D = map(int, input().split())
schedules = []
for _ in range(N):
    schedules.append(input().strip())

# Check which days all people are free
all_free_days = []
for day in range(D):
    all_free = True
    for person in range(N):
        if schedules[person][day] == 'x':
            all_free = False
            break
    all_free_days.append(all_free)

# Find maximum consecutive days where all are free
max_consecutive = 0
current_consecutive = 0

for day in range(D):
    if all_free_days[day]:
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(max_consecutive)