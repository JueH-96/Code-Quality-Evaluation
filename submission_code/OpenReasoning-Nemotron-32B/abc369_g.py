import sys
import heapq
from collections import defaultdict, deque

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	graph = defaultdict(list)
	edges = []
	index = 1
	for i in range(n-1):
		u = int(data[index]); v = int(data[index+1]); l = int(data[index+2])
		index += 3
		graph[u].append((v, l))
		graph[v].append((u, l))
		edges.append((u, v, l))
	
	parent = [0] * (n+1)
	depth = [0] * (n+1)
	in_time = [0] * (n+1)
	out_time = [0] * (n+1)
	timer = 0
	children = [[] for _ in range(n+1)]
	edge_weight_to_parent = [0] * (n+1)
	stack = [1]
	parent[1] = 0
	depth[1] = 0
	order = []
	while stack:
		u = stack.pop()
		order.append(u)
		in_time[u] = timer
		timer += 1
		for v, w in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			depth[v] = depth[u] + w
			edge_weight_to_parent[v] = w
			children[u].append(v)
			stack.append(v)
	
	is_leaf = [len(children[i]) == 0 for i in range(1, n+1)]
	leaves = [i for i in range(1, n+1) if is_leaf[i-1]]
	
	dsu_parent = list(range(n+1))
	dsu_next = [0] * (n+1)
	for i in range(1, n+1):
		dsu_next[i] = parent[i]
	
	def find(x):
		if dsu_parent[x] != x:
			dsu_parent[x] = find(dsu_parent[x])
		return dsu_parent[x]
	
	dist = [0] * (n+1)
	heap = []
	for leaf in leaves:
		dist[leaf] = depth[leaf]
		heapq.heappush(heap, (-dist[leaf], leaf))
	
	total_tree_weight = sum(l for _, _, l in edges)
	ans = [0] * (n+1)
	total_value = 0
	covered_edges = set()
	group_top = [0] * (n+1)
	for i in range(1, n+1):
		group_top[i] = i
	
	for k in range(1, n+1):
		if not heap:
			ans[k] = total_value
			continue
		neg_d, leaf = heapq.heappop(heap)
		d_val = -neg_d
		if d_val != dist[leaf]:
			continue
		total_value += d_val
		ans[k] = total_value
		v = leaf
		path = []
		while v != 0:
			p = parent[v]
			if p == 0:
				break
			if (min(v, p), max(v, p)) in covered_edges:
				break
			covered_edges.add((min(v, p), max(v, p)))
			path.append(v)
			v = p
		if not path:
			continue
		top_node = path[-1]
		for u in path:
			group_top[u] = top_node
			if u == top_node:
				break
			w = edge_weight_to_parent[u]
			dist[u] -= w
			heapq.heappush(heap, (-dist[u], u))
			dsu_parent[u] = top_node
			dsu_next[u] = dsu_next[top_node]
	
	for k in range(1, n+1):
		res = 2 * ans[k]
		print(res)

if __name__ == '__main__':
	main()