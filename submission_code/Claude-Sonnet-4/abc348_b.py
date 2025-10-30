# YOUR CODE HERE
import math

n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(n):
    max_distance = -1
    farthest_id = -1
    
    for j in range(n):
        if i != j:
            # Calculate Euclidean distance
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Update farthest point
            if distance > max_distance or (distance == max_distance and j + 1 < farthest_id):
                max_distance = distance
                farthest_id = j + 1
    
    print(farthest_id)