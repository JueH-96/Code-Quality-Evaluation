import math

def find_pair(N):
    for x in range(1, int(N**(1/3)) + 1):
        y = x - 1
        while True:
            diff = x**3 - y**3
            if diff == N:
                return x, y
            elif diff < N:
                y += 1
            else:
                break
    return -1, -1

N = int(input())
x, y = find_pair(N)
if x == -1 and y == -1:
    print(-1)
else:
    print(x, y)