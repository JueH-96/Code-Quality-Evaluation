import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	total = 0
	index = 1
	for _ in range(n-1):
		a = int(data[index])
		b = int(data[index+1])
		c = int(data[index+2])
		index += 3
		graph[a].append((b, c))
		graph[b].append((a, c))
		total += c

	def bfs(start):
		dist = [-1] * (n+1)
		q = deque()
		q.append(start)
		dist[start] = 0
		farthest = start
		maxd = 0
		while q:
			u = q.popleft()
			if dist[u] > maxd:
				maxd = dist[u]
				farthest = u
			for v, w in graph[u]:
				if dist[v] == -1:
					dist[v] = dist[u] + w
					q.append(v)
		return farthest, maxd

	u, _ = bfs(1)
	v, d = bfs(u)
	ans = 2 * total - d
	print(ans)

if __name__ == "__main__":
	main()