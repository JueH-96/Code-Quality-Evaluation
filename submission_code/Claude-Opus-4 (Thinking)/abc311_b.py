# YOUR CODE HERE
N, D = map(int, input().split())
schedules = []
for _ in range(N):
    schedules.append(input())

max_consecutive = 0
current_consecutive = 0

for day in range(D):
    # Check if all people are free on this day
    if all(schedules[person][day] == 'o' for person in range(N)):
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(max_consecutive)