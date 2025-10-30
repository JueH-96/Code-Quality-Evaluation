# YOUR CODE HERE
N, D = map(int, input().split())
schedules = []
for _ in range(N):
    schedules.append(input().strip())

# Check each day to see if all people are free
all_free = []
for day in range(D):
    free = True
    for person in range(N):
        if schedules[person][day] == 'x':
            free = False
            break
    all_free.append(free)

# Find the longest consecutive sequence of True values
max_consecutive = 0
current_consecutive = 0

for day in range(D):
    if all_free[day]:
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(max_consecutive)