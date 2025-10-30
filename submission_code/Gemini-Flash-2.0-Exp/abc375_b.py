import math

def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    total_cost = 0.0
    
    # From origin to first point
    total_cost += math.sqrt((points[0][0] - 0)**2 + (points[0][1] - 0)**2)

    # From point i to point i+1
    for i in range(n - 1):
        total_cost += math.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)
    
    # From last point to origin
    total_cost += math.sqrt((0 - points[n-1][0])**2 + (0 - points[n-1][1])**2)

    print(total_cost)

solve()