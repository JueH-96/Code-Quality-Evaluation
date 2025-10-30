import math
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
points = []
index = 1

for _ in range(N):
    x = int(data[index])
    y = int(data[index + 1])
    points.append((x, y))
    index += 2

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

total_cost = 0.0
current_point = (0, 0)

for point in points:
    total_cost += calculate_distance(current_point, point)
    current_point = point

total_cost += calculate_distance(current_point, (0, 0))

print(total_cost)