import math

# Read the input
N = int(input())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# Calculate the total cost
total_cost = 0
for i in range(N):
    x1, y1 = 0, 0 if i == 0 else points[i-1]
    x2, y2 = points[i]
    cost = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_cost += cost

# Add the cost to return to the origin
x, y = points[-1]
cost = math.sqrt(x**2 + y**2)
total_cost += cost

# Print the answer
print(f"{total_cost:.15f}")