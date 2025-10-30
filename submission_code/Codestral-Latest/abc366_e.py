import sys
from itertools import product

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
points = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(N)]

min_x = min(x for x, y in points)
max_x = max(x for x, y in points)
min_y = min(y for x, y in points)
max_y = max(y for x, y in points)

def manhattan_distance(x, y, point):
    px, py = point
    return abs(x - px) + abs(y - py)

count = 0
for x in range(min_x, max_x + 1):
    for y in range(min_y, max_y + 1):
        if sum(manhattan_distance(x, y, point) for point in points) <= D:
            count += 1

print(count)