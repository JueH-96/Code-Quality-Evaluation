import math
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
points = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * N + 1, 2)]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

total_cost = 0
for i in range(N):
    total_cost += distance(points[i], points[(i + 1) % N])

print(total_cost)