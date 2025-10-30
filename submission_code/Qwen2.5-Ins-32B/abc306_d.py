# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    idx = 1
    courses = []
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx + 1])
        courses.append((x, y))
        idx += 2

    healthy = [0] * n
    upset = [0] * n

    if courses[0][0] == 0:
        healthy[0] = courses[0][1]
    else:
        upset[0] = courses[0][1]

    for i in range(1, n):
        x, y = courses[i]
        if x == 0:
            healthy[i] = max(healthy[i - 1] + y, healthy[i - 1])
            upset[i] = max(upset[i - 1] + y, upset[i - 1])
        else:
            healthy[i] = max(healthy[i - 1], upset[i - 1] + y)
            upset[i] = max(upset[i - 1], healthy[i - 1] + y)

    print(max(healthy[-1], upset[-1] if courses[-1][0] == 0 else 0))

solve()