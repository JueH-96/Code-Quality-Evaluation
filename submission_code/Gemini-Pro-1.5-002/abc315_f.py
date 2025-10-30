# YOUR CODE HERE
import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    min_s = float('inf')

    for i in range(1 << (n - 2)):
        path = [points[0]]
        skipped = 0
        for j in range(n - 2):
            if (i >> j) & 1:
                path.append(points[j + 1])
            else:
                skipped += 1
        path.append(points[n - 1])

        total_dist = 0
        for k in range(len(path) - 1):
            total_dist += dist(path[k], path[k + 1])

        penalty = 0
        if skipped > 0:
            penalty = 2**(skipped - 1)
        
        min_s = min(min_s, total_dist + penalty)

    print(min_s)

solve()