# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
M = int(input[0])
D = int(input[1])
y = int(input[2])
m = int(input[3])
d = int(input[4])

d += 1
if d > D:
    d = 1
    m += 1
    if m > M:
        m = 1
        y += 1

print(y, m, d)