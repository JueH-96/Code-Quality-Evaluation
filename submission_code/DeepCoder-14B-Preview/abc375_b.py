import math

n = int(input())
points_list = []
for _ in range(n):
    x, y = map(int, input().split())
    points_list.append((x, y))

points = [(0, 0)]
points.extend(points_list)
points.append((0, 0))

total = 0.0
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    dx = x2 - x1
    dy = y2 - y1
    total += math.sqrt(dx ** 2 + dy ** 2)

print("{0:.15f}".format(total))