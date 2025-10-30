import math

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_cost = 0.0

# Start from origin
prev_x, prev_y = 0, 0

# Visit each point in order
for x, y in points:
    total_cost += math.sqrt((prev_x - x)**2 + (prev_y - y)**2)
    prev_x, prev_y = x, y

# Return to origin
total_cost += math.sqrt(prev_x**2 + prev_y**2)

print(total_cost)