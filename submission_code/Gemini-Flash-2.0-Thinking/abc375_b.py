import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

total_cost = 0.0
current_point = (0, 0)

for next_point in points:
    total_cost += distance(current_point, next_point)
    current_point = next_point

total_cost += distance(current_point, (0, 0))

print(total_cost)