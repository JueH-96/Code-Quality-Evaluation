import sys
input = sys.stdin.readline

def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if (dx + dy) % 2 == 0:
        return (dx + dy) // 2
    else:
        return 0

def solve():
    N = int(input())
    points = [tuple(map(int, input().split())) for _ in range(N)]
    points.sort()
    total_dist = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_dist += dist(points[i], points[j])
    print(total_dist)

solve()