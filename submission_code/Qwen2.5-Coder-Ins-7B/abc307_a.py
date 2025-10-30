# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
steps = list(map(int, data[1:]))

total_steps = []
for i in range(N):
    week_steps = sum(steps[i*7:(i+1)*7])
    total_steps.append(week_steps)

print(" ".join(map(str, total_steps)))