# YOUR CODE HERE
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

N = int(input())
points = []

for i in range(N):
    x, y = map(int, input().split())
    points.append((i+1, x, y))

for i, x1, y1 in points:
    max_dist = -1
    farthest_id = -1
    
    for j, x2, y2 in points:
        if i != j:
            dist = distance(x1, y1, x2, y2)
            if dist > max_dist or (dist == max_dist and j < farthest_id):
                max_dist = dist
                farthest_id = j
    
    print(farthest_id)