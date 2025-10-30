# YOUR CODE HERE
import math

# Read number of points
N = int(input())

# Read coordinates
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# For each point, find the farthest point
for i in range(N):
    max_distance = -1
    farthest_point = -1
    
    for j in range(N):
        if i == j:
            continue
        
        # Calculate Euclidean distance
        x1, y1 = points[i]
        x2, y2 = points[j]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        # Update if this is the farthest point so far
        if distance > max_distance:
            max_distance = distance
            farthest_point = j + 1  # Convert to 1-based ID
    
    print(farthest_point)