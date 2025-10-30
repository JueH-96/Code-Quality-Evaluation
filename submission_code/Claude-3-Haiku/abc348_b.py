import math

# Read the number of points
N = int(input())

# Read the coordinates of the points
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# For each point, find the farthest point
for i in range(N):
    max_dist = 0
    farthest_point = 0
    for j in range(N):
        if i != j:
            dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            if dist > max_dist:
                max_dist = dist
                farthest_point = j+1
            elif dist == max_dist and j+1 < farthest_point:
                farthest_point = j+1
    print(farthest_point)