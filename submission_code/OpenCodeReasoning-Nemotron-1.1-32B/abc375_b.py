import math

n = int(input().strip())
prev_x, prev_y = 0, 0
total = 0.0

for _ in range(n):
	x, y = map(int, input().split())
	dx = prev_x - x
	dy = prev_y - y
	total += math.sqrt(dx*dx + dy*dy)
	prev_x, prev_y = x, y

dx = prev_x - 0
dy = prev_y - 0
total += math.sqrt(dx*dx + dy*dy)

print("{:.20f}".format(total))