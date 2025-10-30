# YOUR CODE HERE
import sys
input = sys.stdin.read

def next_day(M, D, y, m, d):
    if d < D:
        return y, m, d + 1
    elif m < M:
        return y, m + 1, 1
    else:
        return y + 1, 1, 1

data = input().strip().split()
M = int(data[0])
D = int(data[1])
y = int(data[2])
m = int(data[3])
d = int(data[4])

y_next, m_next, d_next = next_day(M, D, y, m, d)
print(y_next, m_next, d_next)