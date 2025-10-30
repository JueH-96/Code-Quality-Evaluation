import math

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

total = 0.0

# Distance from origin to first point
total += math.hypot(points[0][0], points[0][1])

# Distances between consecutive points
for i in range(1, n):
    prev = points[i-1]
    curr = points[i]
    total += math.hypot(curr[0] - prev[0], curr[1] - prev[1])

# Distance from last point back to origin
total += math.hypot(points[-1][0], points[-1][1])

# Output with sufficient precision
print("{0:.20f}".format(total))