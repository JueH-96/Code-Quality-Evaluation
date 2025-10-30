import sys
import math

def solve():
    N = int(sys.stdin.readline().strip())
    points = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    points.sort(key=lambda x: (x[0], x[1]))

    penalty = 0
    total_distance = 0
    for i in range(1, N):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        total_distance += distance
        if i > 1:
            penalty += 2**(i-1)

    total_distance += penalty
    print(total_distance)

solve()