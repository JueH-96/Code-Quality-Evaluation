import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it)); m = int(next(it)); K = int(next(it))
	edges = []
	graph = [[] for _ in range(n+1)]
	for i in range(m):
		u = int(next(it)); v = int(next(it))
		edges.append((u, v))
		graph[u].append((v, i))
	
	dist = [-1] * (n+1)
	dist[1] = 0
	q = deque([1])
	while q:
		u = q.popleft()
		for (v, e_idx) in graph[u]:
			if dist[v] == -1:
				dist[v] = dist[u] + 1
				q.append(v)
	min_path_length = dist[n]
	
	paths = []
	visited = [False] * (n+1)
	current_path_edges = []

	def dfs(u):
		if u == n:
			paths.append(list(current_path_edges))
			return
		visited[u] = True
		for (v, e_idx) in graph[u]:
			if not visited[v]:
				current_path_edges.append(e_idx)
				dfs(v)
				current_path_edges.pop()
		visited[u] = False

	dfs(1)
	
	if not paths:
		print(0)
		return

	edge_to_paths = [[] for _ in range(m)]
	for p_idx, path in enumerate(paths):
		for e_idx in path:
			edge_to_paths[e_idx].append(p_idx)
			
	d_max = min(K, min_path_length)
	
	def can_cover(d, K, paths, edge_to_paths):
		if d == 0:
			return True
		n_paths = len(paths)
		coverage = [0] * n_paths
		selected_edges = set()
		while min(coverage) < d and len(selected_edges) < K:
			best_edge = None
			best_count = -1
			for e_idx in range(m):
				if e_idx in selected_edges:
					continue
				count = 0
				for p_idx in edge_to_paths[e_idx]:
					if coverage[p_idx] < d:
						count += 1
				if count > best_count:
					best_count = count
					best_edge = e_idx
			if best_count == 0:
				break
			selected_edges.add(best_edge)
			for p_idx in edge_to_paths[best_edge]:
				coverage[p_idx] += 1
		return min(coverage) >= d

	for d in range(d_max, -1, -1):
		if can_cover(d, K, paths, edge_to_paths):
			print(d)
			return
	print(0)

if __name__ == "__main__":
	main()