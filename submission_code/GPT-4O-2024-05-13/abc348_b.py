import math

def find_farthest_points(N, points):
    results = []
    for i in range(N):
        max_distance = -1
        farthest_point_id = -1
        for j in range(N):
            if i != j:
                distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_point_id = j + 1
                elif distance == max_distance:
                    farthest_point_id = min(farthest_point_id, j + 1)
        results.append(farthest_point_id)
    return results

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
points = []
index = 1
for i in range(N):
    x = int(data[index])
    y = int(data[index + 1])
    points.append((x, y))
    index += 2

# Find and print the farthest points
results = find_farthest_points(N, points)
for result in results:
    print(result)