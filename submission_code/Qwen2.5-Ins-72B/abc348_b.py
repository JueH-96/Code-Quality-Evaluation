import math

# Read the number of points
N = int(input())

# Read the coordinates of the points
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# For each point, find the farthest point and print its ID
for i in range(N):
    max_distance = 0
    farthest_point_id = i + 1
    for j in range(N):
        if i != j:
            d = distance(points[i], points[j])
            if d > max_distance or (d == max_distance and j + 1 < farthest_point_id):
                max_distance = d
                farthest_point_id = j + 1
    print(farthest_point_id)