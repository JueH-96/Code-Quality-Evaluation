def dist(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if (dx % 2) != (dy % 2):
        return 0
    return max(dx, dy)

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_dist = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        total_dist += dist(points[i], points[j])

print(total_dist)