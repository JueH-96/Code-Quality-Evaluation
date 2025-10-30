import math

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

total = 0.0
prev_x, prev_y = 0, 0

for x, y in points:
    dx = x - prev_x
    dy = y - prev_y
    total += math.hypot(dx, dy)
    prev_x, prev_y = x, y

# Add the return to origin
dx = -prev_x
dy = -prev_y
total += math.hypot(dx, dy)

print("{0:.20f}".format(total))