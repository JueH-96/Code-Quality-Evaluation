# YOUR CODE HERE
import math

N = int(input())
points = [(0, 0)]  # Start with origin
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))
points.append((0, 0))  # Return to origin

total_distance = 0
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    total_distance += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

print(total_distance)