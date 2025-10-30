import math

def main():
	n = int(input().strip())
	points = []
	for _ in range(n):
		x, y = map(int, input().split())
		points.append((x, y))
	
	total = 0.0
	prev_x, prev_y = 0, 0
	for x, y in points:
		dx = x - prev_x
		dy = y - prev_y
		total += math.sqrt(dx*dx + dy*dy)
		prev_x, prev_y = x, y
	
	dx = -prev_x
	dy = -prev_y
	total += math.sqrt(dx*dx + dy*dy)
	
	print("{:.20f}".format(total))

if __name__ == "__main__":
	main()