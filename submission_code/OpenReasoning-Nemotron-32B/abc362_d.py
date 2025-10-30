import sys
import heapq

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	graph = [[] for _ in range(n)]
	index = 2 + n
	for _ in range(m):
		u = int(data[index])
		v = int(data[index+1])
		B = int(data[index+2])
		index += 3
		u0 = u - 1
		v0 = v - 1
		graph[u0].append((v0, B))
		graph[v0].append((u0, B))
	
	INF = 10**18
	dist = [INF] * n
	dist[0] = A[0]
	heap = [(A[0], 0)]
	
	while heap:
		d, u = heapq.heappop(heap)
		if d != dist[u]:
			continue
		for v, B in graph[u]:
			new_d = d + B + A[v]
			if new_d < dist[v]:
				dist[v] = new_d
				heapq.heappush(heap, (new_d, v))
	
	result = []
	for i in range(1, n):
		result.append(str(dist[i]))
	print(" ".join(result))

if __name__ == "__main__":
	main()