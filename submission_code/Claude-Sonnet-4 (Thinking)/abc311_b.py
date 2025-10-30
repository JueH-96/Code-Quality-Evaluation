# Read input
N, D = map(int, input().split())
schedules = []
for i in range(N):
    schedules.append(input().strip())

# Find maximum consecutive days where all are free
max_consecutive = 0
current_consecutive = 0

for day in range(D):
    # Check if all people are free on this day
    all_free = True
    for person in range(N):
        if schedules[person][day] == 'x':
            all_free = False
            break
    
    if all_free:
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(max_consecutive)