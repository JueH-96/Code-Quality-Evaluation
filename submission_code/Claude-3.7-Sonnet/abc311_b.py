# YOUR CODE HERE
N, D = map(int, input().split())
schedules = []
for _ in range(N):
    schedules.append(input().strip())

max_streak = 0
current_streak = 0

for day in range(D):
    if all(schedules[person][day] == 'o' for person in range(N)):
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        current_streak = 0

print(max_streak)