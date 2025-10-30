# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
steps = list(map(int, input[1:]))

weekly_steps = []
for i in range(N):
    weekly_sum = sum(steps[i*7:(i+1)*7])
    weekly_steps.append(weekly_sum)

print(" ".join(map(str, weekly_steps)))