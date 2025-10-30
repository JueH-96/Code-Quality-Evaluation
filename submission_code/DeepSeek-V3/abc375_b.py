import math

def calculate_total_cost(N, points):
    total_cost = 0.0
    current_x, current_y = 0, 0
    for x, y in points:
        dx = x - current_x
        dy = y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        total_cost += distance
        current_x, current_y = x, y
    # Return to origin
    dx = 0 - current_x
    dy = 0 - current_y
    distance = math.sqrt(dx*dx + dy*dy)
    total_cost += distance
    return total_cost

# Read input
N = int(input())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# Calculate total cost
total_cost = calculate_total_cost(N, points)

# Print the result with sufficient precision
print("{0:.20f}".format(total_cost))