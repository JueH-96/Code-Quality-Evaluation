import heapq
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	x = int(data[2])
	graph0 = [[] for _ in range(n)]
	graph1 = [[] for _ in range(n)]
	
	index = 3
	for _ in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		u -= 1
		v -= 1
		graph0[u].append(v)
		graph1[v].append(u)
	
	total_nodes = 2 * n
	INF = 10**18
	dist = [INF] * total_nodes
	heap = []
	dist[0] = 0
	heapq.heappush(heap, (0, 0))
	
	while heap:
		d, node = heapq.heappop(heap)
		if d != dist[node]:
			continue
			
		if node < n:
			for neighbor in graph0[node]:
				next_node = neighbor
				new_d = d + 1
				if new_d < dist[next_node]:
					dist[next_node] = new_d
					heapq.heappush(heap, (new_d, next_node))
					
			next_node = node + n
			new_d = d + x
			if new_d < dist[next_node]:
				dist[next_node] = new_d
				heapq.heappush(heap, (new_d, next_node))
				
		else:
			u = node - n
			for neighbor in graph1[u]:
				next_node = neighbor + n
				new_d = d + 1
				if new_d < dist[next_node]:
					dist[next_node] = new_d
					heapq.heappush(heap, (new_d, next_node))
					
			next_node = u
			new_d = d + x
			if new_d < dist[next_node]:
				dist[next_node] = new_d
				heapq.heappush(heap, (new_d, next_node))
				
	ans = min(dist[n-1], dist[2*n-1])
	print(ans)

if __name__ == "__main__":
	main()