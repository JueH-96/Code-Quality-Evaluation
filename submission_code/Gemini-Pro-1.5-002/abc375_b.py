import math

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_cost = 0
current_x, current_y = 0, 0
for x, y in points:
    cost = math.sqrt((current_x - x)**2 + (current_y - y)**2)
    total_cost += cost
    current_x, current_y = x, y

cost = math.sqrt((current_x - 0)**2 + (current_y - 0)**2)
total_cost += cost

print(total_cost)