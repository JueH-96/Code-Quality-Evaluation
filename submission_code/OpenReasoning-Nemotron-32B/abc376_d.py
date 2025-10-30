import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	graph = [[] for _ in range(n+1)]
	in_edges_1 = set()
	
	for _ in range(m):
		a = int(next(it))
		b = int(next(it))
		if b == 1:
			in_edges_1.add(a)
		graph[a].append(b)
	
	dist = [-1] * (n+1)
	q = deque()
	dist[1] = 0
	q.append(1)
	
	while q:
		u = q.popleft()
		for v in graph[u]:
			if dist[v] == -1:
				dist[v] = dist[u] + 1
				q.append(v)
	
	ans = float('inf')
	for u in in_edges_1:
		if dist[u] != -1:
			ans = min(ans, dist[u] + 1)
	
	print(-1 if ans == float('inf') else ans)

if __name__ == '__main__':
	main()