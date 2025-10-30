import math

N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y, i + 1))

# Calculate the distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# For each point, find the farthest point
for i in range(N):
    max_distance = 0
    farthest_point = 0
    for j in range(N):
        if i == j:
            continue
        d = distance(points[i], points[j])
        if d > max_distance:
            max_distance = d
            farthest_point = points[j][2]
        elif d == max_distance and points[j][2] < farthest_point:
            farthest_point = points[j][2]
    print(farthest_point)