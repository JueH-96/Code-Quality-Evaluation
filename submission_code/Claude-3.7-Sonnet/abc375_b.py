import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Read input
N = int(input())
points = []
for _ in range(N):
    X, Y = map(int, input().split())
    points.append((X, Y))

# Calculate total cost
total_cost = 0

# From origin to first point
total_cost += calculate_distance(0, 0, points[0][0], points[0][1])

# Between consecutive points
for i in range(N-1):
    x1, y1 = points[i]
    x2, y2 = points[i+1]
    total_cost += calculate_distance(x1, y1, x2, y2)

# From last point back to origin
total_cost += calculate_distance(points[-1][0], points[-1][1], 0, 0)

# Print the answer
print(total_cost)