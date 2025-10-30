def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	points = []
	for i in range(1, n + 1):
		x, y = map(int, data[i].split())
		points.append((x, y))
	
	results = []
	for i in range(n):
		max_d = -1
		candidate = -1
		for j in range(n):
			if i == j:
				continue
			x1, y1 = points[i]
			x2, y2 = points[j]
			dx = x1 - x2
			dy = y1 - y2
			d_sq = dx * dx + dy * dy
			if d_sq > max_d:
				max_d = d_sq
				candidate = j
			elif d_sq == max_d:
				if j < candidate:
					candidate = j
		results.append(str(candidate + 1))
	
	print("
".join(results))

if __name__ == "__main__":
	main()