import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	index = 1
	edges = []
	for i in range(n-1):
		a = int(data[index]); b = int(data[index+1]); index += 2
		graph[a].append(b)
		graph[b].append(a)
		edges.append((a, b))
	
	parent0 = [0] * (n+1)
	depth = [0] * (n+1)
	queue = deque([1])
	depth[1] = 0
	parent0[1] = 0
	while queue:
		u = queue.popleft()
		for v in graph[u]:
			if v == parent0[u]:
				continue
			parent0[v] = u
			depth[v] = depth[u] + 1
			queue.append(v)
			
	k = 0
	while (1 << k) <= n:
		k += 1
	k += 1
	parent_table = [[0] * (n+1) for _ in range(k)]
	for i in range(1, n+1):
		parent_table[0][i] = parent0[i]
		
	for i in range(1, k):
		for j in range(1, n+1):
			if parent_table[i-1][j] != 0:
				parent_table[i][j] = parent_table[i-1][parent_table[i-1][j]]
			else:
				parent_table[i][j] = 0
				
	def lca(a, b):
		if depth[a] < depth[b]:
			a, b = b, a
		d = depth[a] - depth[b]
		bit = 0
		temp = d
		while temp:
			if temp & 1:
				a = parent_table[bit][a]
			temp //= 2
			bit += 1
		if a == b:
			return a
		for i in range(k-1, -1, -1):
			if parent_table[i][a] != parent_table[i][b]:
				a = parent_table[i][a]
				b = parent_table[i][b]
		return parent0[a]
	
	def dist(u, v):
		w = lca(u, v)
		return depth[u] + depth[v] - 2 * depth[w]
	
	curr_degree = [0] * (n+1)
	for i in range(1, n+1):
		curr_degree[i] = len(graph[i])
		
	leaves = set()
	for i in range(1, n+1):
		if curr_degree[i] == 1:
			leaves.add(i)
			
	operations = []
	
	def remove_leaf(x, leaves, curr_degree, graph):
		if curr_degree[x] == 0:
			return
		for neighbor in graph[x]:
			if curr_degree[neighbor] > 0:
				break
		curr_degree[x] = 0
		if x in leaves:
			leaves.remove(x)
		curr_degree[neighbor] -= 1
		if curr_degree[neighbor] == 1:
			leaves.add(neighbor)
			
	for _ in range(n//2):
		if len(leaves) == 0:
			break
		if len(leaves) == 2:
			u, v = list(leaves)
			operations.append((u, v))
			remove_leaf(u, leaves, curr_degree, graph)
			remove_leaf(v, leaves, curr_degree, graph)
		else:
			if len(leaves) <= 1000:
				a = next(iter(leaves))
				b = a
				max_da = -1
				for leaf in leaves:
					d_val = dist(a, leaf)
					if d_val > max_da:
						max_da = d_val
						b = leaf
				c = b
				max_db = -1
				for leaf in leaves:
					d_val = dist(b, leaf)
					if d_val > max_db:
						max_db = d_val
						c = leaf
				u, v = b, c
			else:
				a = max(leaves, key=lambda x: depth[x])
				b = None
				min_depth_val = 10**9
				for leaf in leaves:
					if leaf == a:
						continue
					if depth[leaf] < min_depth_val:
						min_depth_val = depth[leaf]
						b = leaf
				if b is None:
					b = a
				u, v = a, b
			operations.append((u, v))
			remove_leaf(u, leaves, curr_degree, graph)
			remove_leaf(v, leaves, curr_degree, graph)
			
	for op in operations:
		print(f"{op[0]} {op[1]}")
		
if __name__ == "__main__":
	main()