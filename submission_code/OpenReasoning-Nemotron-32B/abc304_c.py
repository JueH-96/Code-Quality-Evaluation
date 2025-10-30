import collections

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, d = map(int, data[0].split())
	points = []
	for i in range(1, 1 + n):
		x, y = map(int, data[i].split())
		points.append((x, y))
	
	graph = [[] for _ in range(n)]
	d_sq = d * d
	
	for i in range(n):
		for j in range(i + 1, n):
			dx = points[i][0] - points[j][0]
			dy = points[i][1] - points[j][1]
			dist_sq = dx * dx + dy * dy
			if dist_sq <= d_sq:
				graph[i].append(j)
				graph[j].append(i)
	
	visited = [False] * n
	queue = collections.deque()
	queue.append(0)
	visited[0] = True
	
	while queue:
		node = queue.popleft()
		for neighbor in graph[node]:
			if not visited[neighbor]:
				visited[neighbor] = True
				queue.append(neighbor)
	
	for i in range(n):
		print("Yes" if visited[i] else "No")

if __name__ == "__main__":
	main()