import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
steps = list(map(int, data[1:]))

weekly_steps = []
for i in range(N):
    week_sum = sum(steps[i*7:(i+1)*7])
    weekly_steps.append(week_sum)

print(" ".join(map(str, weekly_steps)))