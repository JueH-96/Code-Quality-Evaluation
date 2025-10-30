import math

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

prev_x, prev_y = 0.0, 0.0
total = 0.0

for x, y in points:
    dx = x - prev_x
    dy = y - prev_y
    total += math.hypot(dx, dy)
    prev_x, prev_y = x, y

# Return to origin
dx = 0 - prev_x
dy = 0 - prev_y
total += math.hypot(dx, dy)

print("{0:.20f}".format(total))