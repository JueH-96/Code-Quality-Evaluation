import sys
import heapq

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	A = [int(next(it)) for _ in range(n)]
	graph = [[] for _ in range(n)]
	
	for _ in range(m):
		u = int(next(it)) - 1
		v = int(next(it)) - 1
		w = int(next(it))
		graph[u].append((v, w + A[v]))
		graph[v].append((u, w + A[u]))
	
	INF = 10**18
	dist = [INF] * n
	dist[0] = A[0]
	heap = [(A[0], 0)]
	
	while heap:
		d, u = heapq.heappop(heap)
		if d != dist[u]:
			continue
		for v, weight in graph[u]:
			new_d = d + weight
			if new_d < dist[v]:
				dist[v] = new_d
				heapq.heappush(heap, (new_d, v))
	
	print(" ".join(str(dist[i]) for i in range(1, n)))

if __name__ == "__main__":
	main()