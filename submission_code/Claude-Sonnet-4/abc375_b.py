import math

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

total_cost = 0.0
current_pos = (0, 0)  # Start at origin

# Visit each point in order
for point in points:
    total_cost += distance(current_pos, point)
    current_pos = point

# Return to origin
total_cost += distance(current_pos, (0, 0))

print(total_cost)