# YOUR CODE HERE
def dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if (dx - dy) % 2 != 0:
        return 0
    else:
        return max(dx, dy)

N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

total_dist = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        x1, y1 = points[i]
        x2, y2 = points[j]
        total_dist += dist(x1, y1, x2, y2)

print(total_dist)