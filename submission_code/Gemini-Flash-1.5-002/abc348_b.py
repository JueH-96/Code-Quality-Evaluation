# YOUR CODE HERE
import math

n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(n):
    max_dist = -1
    farthest_point_id = -1
    for j in range(n):
        if i == j:
            continue
        dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
        if dist > max_dist:
            max_dist = dist
            farthest_point_id = j + 1
        elif dist == max_dist:
            farthest_point_id = min(farthest_point_id, j + 1)
    print(farthest_point_id)