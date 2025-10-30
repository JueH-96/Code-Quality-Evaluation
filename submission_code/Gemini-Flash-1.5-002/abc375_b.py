import math

n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_cost = 0
prev_x, prev_y = 0, 0
for x, y in points:
    cost = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    total_cost += cost
    prev_x, prev_y = x, y

cost = math.sqrt((prev_x - 0)**2 + (prev_y - 0)**2)
total_cost += cost

print(total_cost)