import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	n, d = map(int, data[0].split())
	points = []
	for i in range(1, n + 1):
		x, y = map(int, data[i].split())
		points.append((x, y))
	
	infected = [False] * n
	queue = deque()
	infected[0] = True
	queue.append(0)
	d_sq = d * d
	
	while queue:
		u = queue.popleft()
		for v in range(n):
			if not infected[v]:
				x1, y1 = points[u]
				x2, y2 = points[v]
				dx = x1 - x2
				dy = y1 - y2
				dist_sq = dx * dx + dy * dy
				if dist_sq <= d_sq:
					infected[v] = True
					queue.append(v)
	
	for i in range(n):
		print("Yes" if infected[i] else "No")

if __name__ == '__main__':
	main()