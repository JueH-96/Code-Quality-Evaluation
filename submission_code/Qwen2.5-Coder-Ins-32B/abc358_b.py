# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
A = int(input[1])
T = list(map(int, input[2:]))

finish_times = []
current_time = 0

for t in T:
    if t > current_time:
        current_time = t
    current_time += A
    finish_times.append(current_time)

for time in finish_times:
    print(time)