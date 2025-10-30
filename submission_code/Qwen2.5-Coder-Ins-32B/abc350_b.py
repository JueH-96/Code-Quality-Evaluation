# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
Q = int(input[1])
T = list(map(int, input[2:]))

teeth = set(range(1, N + 1))

for t in T:
    if t in teeth:
        teeth.remove(t)
    else:
        teeth.add(t)

print(len(teeth))