# YOUR CODE HERE
def next_day(M, D, y, m, d):
    if d < D:
        return y, m, d + 1
    elif m < M:
        return y, m + 1, 1
    else:
        return y + 1, 1, 1

import sys
input = sys.stdin.read
data = input().split()
M = int(data[0])
D = int(data[1])
y = int(data[2])
m = int(data[3])
d = int(data[4])

result = next_day(M, D, y, m, d)
print(result[0], result[1], result[2])