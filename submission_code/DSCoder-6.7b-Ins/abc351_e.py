from collections import deque

def dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)

def solve():
    N = int(input())
    points = [tuple(map(int, input().split())) for _ in range(N)]
    total_dist = 0
    for i in range(N):
        for j in range(i+1, N):
            total_dist += dist(*points[i], *points[j])
    print(total_dist)

solve()