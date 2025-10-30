# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

M = int(data[0])
D = int(data[1])
y = int(data[2])
m = int(data[3])
d = int(data[4])

if d == D:
    if m == M:
        y += 1
        m = 1
        d = 1
    else:
        m += 1
        d = 1
else:
    d += 1

print(y, m, d)