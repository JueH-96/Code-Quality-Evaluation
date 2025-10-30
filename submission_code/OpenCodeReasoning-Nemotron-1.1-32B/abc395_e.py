import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	X = int(data[2])
	edges = []
	index = 3
	graph0 = [[] for _ in range(n)]
	graph1 = [[] for _ in range(n)]
	
	for i in range(m):
		u = int(data[index]); v = int(data[index+1]); index += 2
		u0 = u - 1
		v0 = v - 1
		graph0[u0].append(v0)
		graph1[v0].append(u0)
	
	INF = 10**18
	dist0 = [INF] * n
	dist1 = [INF] * n
	heap = []
	dist0[0] = 0
	heapq.heappush(heap, (0, 0, 0))
	
	while heap:
		cost, u, p = heapq.heappop(heap)
		if p == 0:
			if cost != dist0[u]:
				continue
		else:
			if cost != dist1[u]:
				continue
		
		new_cost_rev = cost + X
		new_p = 1 - p
		if new_p == 0:
			if new_cost_rev < dist0[u]:
				dist0[u] = new_cost_rev
				heapq.heappush(heap, (new_cost_rev, u, new_p))
		else:
			if new_cost_rev < dist1[u]:
				dist1[u] = new_cost_rev
				heapq.heappush(heap, (new_cost_rev, u, new_p))
				
		if p == 0:
			for v in graph0[u]:
				new_cost_move = cost + 1
				if new_cost_move < dist0[v]:
					dist0[v] = new_cost_move
					heapq.heappush(heap, (new_cost_move, v, 0))
		else:
			for v in graph1[u]:
				new_cost_move = cost + 1
				if new_cost_move < dist1[v]:
					dist1[v] = new_cost_move
					heapq.heappush(heap, (new_cost_move, v, 1))
					
	ans = min(dist0[n-1], dist1[n-1])
	print(ans)

if __name__ == "__main__":
	main()