import math

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

total = 0.0
prev_x, prev_y = 0, 0

for x, y in points:
    dx = x - prev_x
    dy = y - prev_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    total += distance
    prev_x, prev_y = x, y

# Add the distance back to origin
dx = 0 - prev_x
dy = 0 - prev_y
distance = math.sqrt(dx ** 2 + dy ** 2)
total += distance

print("{0:.20f}".format(total))