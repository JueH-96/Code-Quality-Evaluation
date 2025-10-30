import sys
data = sys.stdin.read().split()
N = int(data[0])
D = int(data[1])
schedules = data[2:2+N]
max_streak = 0
current_streak = 0
for day in range(D):
    if all(s[day] == 'o' for s in schedules):
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        current_streak = 0
print(max_streak)