import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def solve():
    # Read input
    N = int(input())
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    # Calculate total distance
    total_dist = 0.0
    
    # Distance from origin to first point
    total_dist += distance(0, 0, points[0][0], points[0][1])
    
    # Distance between consecutive points
    for i in range(N-1):
        total_dist += distance(points[i][0], points[i][1], 
                             points[i+1][0], points[i+1][1])
    
    # Distance from last point back to origin
    total_dist += distance(points[-1][0], points[-1][1], 0, 0)
    
    print(total_dist)

solve()