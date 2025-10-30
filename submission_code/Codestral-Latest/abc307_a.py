# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
steps = list(map(int, data[1:]))

weekly_steps = []
for i in range(N):
    week_start = i * 7
    week_end = week_start + 7
    weekly_steps.append(sum(steps[week_start:week_end]))

print(" ".join(map(str, weekly_steps)))