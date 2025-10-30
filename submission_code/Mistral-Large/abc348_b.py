import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
points = []

index = 1
for i in range(N):
    X = int(data[index])
    Y = int(data[index + 1])
    points.append((X, Y))
    index += 2

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

for i in range(N):
    max_distance = -1
    farthest_point_id = -1
    for j in range(N):
        if i != j:
            distance = euclidean_distance(points[i], points[j])
            if distance > max_distance or (distance == max_distance and (farthest_point_id == -1 or j < farthest_point_id)):
                max_distance = distance
                farthest_point_id = j
    print(farthest_point_id + 1)