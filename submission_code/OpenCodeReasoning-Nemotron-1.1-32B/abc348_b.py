def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	points = []
	for i in range(1, n + 1):
		x, y = map(int, data[i].split())
		points.append((x, y, i))
	
	results = [0] * n
	for i in range(n):
		x1, y1, id1 = points[i]
		max_d_sq = -1
		candidate_id = None
		for j in range(n):
			if i == j:
				continue
			x2, y2, id2 = points[j]
			dx = x1 - x2
			dy = y1 - y2
			d_sq = dx * dx + dy * dy
			if d_sq > max_d_sq:
				max_d_sq = d_sq
				candidate_id = id2
			elif d_sq == max_d_sq:
				if id2 < candidate_id:
					candidate_id = id2
		results[i] = candidate_id
	
	for res in results:
		print(res)

if __name__ == "__main__":
	main()