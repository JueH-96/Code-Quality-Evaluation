import sys
import math

# Read input
N = int(sys.stdin.readline().strip())
points = []
for _ in range(N):
    x, y = map(int, sys.stdin.readline().strip().split())
    points.append((x, y))

# Function to calculate Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Find the farthest point for each point
results = []
for i in range(N):
    max_distance = 0
    farthest_point_id = i + 1
    for j in range(N):
        if i != j:
            dist = distance(points[i], points[j])
            if dist > max_distance:
                max_distance = dist
                farthest_point_id = j + 1
            elif dist == max_distance:
                farthest_point_id = min(farthest_point_id, j + 1)
    results.append(farthest_point_id)

# Print results
for result in results:
    print(result)