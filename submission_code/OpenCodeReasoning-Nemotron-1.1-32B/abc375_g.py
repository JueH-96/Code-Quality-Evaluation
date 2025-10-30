import sys
import heapq
from collections import deque, defaultdict

def main():
	data = sys.stdin.read().split()
	if not data: 
		return
	it = iter(data)
	n = int(next(it)); m = int(next(it))
	edges = []
	graph = [[] for _ in range(n+1)]
	
	for i in range(m):
		u = int(next(it)); v = int(next(it)); c = int(next(it))
		edges.append((u, v, c))
		graph[u].append((v, c))
		graph[v].append((u, c))
	
	comp1 = set()
	q = deque([1])
	comp1.add(1)
	while q:
		u = q.popleft()
		for v, c in graph[u]:
			if v not in comp1:
				comp1.add(v)
				q.append(v)
				
	dist1 = [10**18] * (n+1)
	dp1 = [0] * (n+1)
	heap = []
	heapq.heappush(heap, (0, 1))
	dist1[1] = 0
	dp1[1] = 1
	
	while heap:
		d, u = heapq.heappop(heap)
		if d != dist1[u]:
			continue
		for v, c in graph[u]:
			nd = d + c
			if nd < dist1[v]:
				dist1[v] = nd
				dp1[v] = dp1[u]
				heapq.heappush(heap, (nd, v))
			elif nd == dist1[v]:
				dp1[v] += dp1[u]
				
	distN = [10**18] * (n+1)
	dpN = [0] * (n+1)
	heap = []
	heapq.heappush(heap, (0, n))
	distN[n] = 0
	dpN[n] = 1
	
	while heap:
		d, u = heapq.heappop(heap)
		if d != distN[u]:
			continue
		for v, c in graph[u]:
			nd = d + c
			if nd < distN[v]:
				distN[v] = nd
				dpN[v] = dpN[u]
				heapq.heappush(heap, (nd, v))
			elif nd == distN[v]:
				dpN[v] += dpN[u]
				
	total_shortest = dist1[n]
	
	comp_graph = [[] for _ in range(n+1)]
	for u in comp1:
		for v, c in graph[u]:
			if v in comp1:
				comp_graph[u].append(v)
				
	disc = [-1] * (n+1)
	low = [-1] * (n+1)
	parent = [-1] * (n+1)
	next_index = [0] * (n+1)
	time = 0
	bridge_set = set()
	
	stack = []
	disc[1] = time
	low[1] = time
	time += 1
	stack.append(1)
	
	while stack:
		u = stack[-1]
		if next_index[u] < len(comp_graph[u]):
			v = comp_graph[u][next_index[u]]
			next_index[u] += 1
			if v == parent[u]:
				continue
			if disc[v] == -1:
				parent[v] = u
				disc[v] = time
				low[v] = time
				time += 1
				stack.append(v)
			else:
				low[u] = min(low[u], disc[v])
		else:
			stack.pop()
			p = parent[u]
			if p != -1:
				low[p] = min(low[p], low[u])
				if low[u] > disc[p]:
					a, b = min(p, u), max(p, u)
					bridge_set.add((a, b))
					
	count_edges = defaultdict(int)
	for u in comp1:
		for v, c in graph[u]:
			if v in comp1 and u < v:
				if dist1[u] + c + distN[v] == total_shortest:
					key = (dist1[u], dist1[v])
					count_edges[key] += 1
				if dist1[v] + c + distN[u] == total_shortest:
					key = (dist1[v], dist1[u])
					count_edges[key] += 1
					
	out_lines = []
	for (u, v, c) in edges:
		if u not in comp1 or v not in comp1:
			out_lines.append("No")
		else:
			cand1 = dist1[u] + c + distN[v]
			cand2 = dist1[v] + c + distN[u]
			if cand1 == total_shortest or cand2 == total_shortest:
				if cand1 == total_shortest:
					key = (dist1[u], dist1[v])
					if count_edges[key] == 1:
						out_lines.append("Yes")
					else:
						out_lines.append("No")
				else:
					key = (dist1[v], dist1[u])
					if count_edges[key] == 1:
						out_lines.append("Yes")
					else:
						out_lines.append("No")
			else:
				a, b = min(u, v), max(u, v)
				if (a, b) in bridge_set:
					out_lines.append("Yes")
				else:
					out_lines.append("No")
					
	for line in out_lines:
		print(line)

if __name__ == "__main__":
	main()