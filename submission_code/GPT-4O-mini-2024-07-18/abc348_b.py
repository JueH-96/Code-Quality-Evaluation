# YOUR CODE HERE
import sys
import math

def find_farthest_points():
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    points = []
    
    for i in range(1, N + 1):
        x, y = map(int, data[i].split())
        points.append((x, y, i))  # Store (x, y, ID)
    
    results = []
    
    for i in range(N):
        x1, y1, id1 = points[i]
        max_distance = -1
        farthest_id = None
        
        for j in range(N):
            if i == j:
                continue
            x2, y2, id2 = points[j]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            
            if distance > max_distance:
                max_distance = distance
                farthest_id = id2
            elif distance == max_distance:
                farthest_id = min(farthest_id, id2)
        
        results.append(farthest_id)
    
    for result in results:
        print(result)

find_farthest_points()