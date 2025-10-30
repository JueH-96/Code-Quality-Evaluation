import collections
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	in_1 = []
	
	index = 2
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		graph[a].append(b)
		if b == 1:
			in_1.append(a)
	
	INF = 10**9
	dist = [INF] * (n+1)
	dist[1] = 0
	q = collections.deque([1])
	
	while q:
		u = q.popleft()
		for v in graph[u]:
			if dist[v] == INF:
				dist[v] = dist[u] + 1
				q.append(v)
	
	ans = INF
	for u in in_1:
		if dist[u] < INF:
			candidate = dist[u] + 1
			if candidate < ans:
				ans = candidate
				
	print(-1 if ans == INF else ans)

if __name__ == "__main__":
	main()