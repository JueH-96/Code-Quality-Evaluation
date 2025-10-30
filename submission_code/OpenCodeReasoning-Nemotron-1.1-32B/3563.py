from typing import List

class Solution:
	def maxScore(self, grid: List[List[int]]) -> int:
		n = len(grid)
		distinct_vals = sorted(set(val for row in grid for val in row))
		m = len(distinct_vals)
		total_nodes = n + m + 2
		source = 0
		sink = total_nodes - 1
		
		graph = [[] for _ in range(total_nodes)]
		
		def add_edge(u, v, cap, cost):
			graph[u].append([v, cap, cost, len(graph[v])])
			graph[v].append([u, 0, -cost, len(graph[u]) - 1])
		
		for i in range(n):
			add_edge(source, i + 1, 1, 0)
		
		val_to_index = {val: idx for idx, val in enumerate(distinct_vals)}
		
		for i in range(n):
			row_vals = set(grid[i])
			for val in row_vals:
				j = val_to_index[val]
				add_edge(i + 1, n + 1 + j, 1, -val)
		
		for j in range(m):
			add_edge(n + 1 + j, sink, 1, 0)
		
		total_flow = 0
		total_cost = 0
		best_score = 0
		
		for _ in range(n):
			dist = [10**9] * total_nodes
			parent = [-1] * total_nodes
			dist[source] = 0
			
			for _ in range(total_nodes - 1):
				updated = False
				for u in range(total_nodes):
					if dist[u] == 10**9:
						continue
					for idx, edge in enumerate(graph[u]):
						if edge[1] > 0:
							new_dist = dist[u] + edge[2]
							if new_dist < dist[edge[0]]:
								dist[edge[0]] = new_dist
								parent[edge[0]] = (u, idx)
								updated = True
				if not updated:
					break
			
			if dist[sink] == 10**9:
				break
			
			f = 1
			total_flow += f
			total_cost += dist[sink]
			
			cur = sink
			while cur != source:
				u, idx = parent[cur]
				edge = graph[u][idx]
				edge[1] -= f
				rev_edge = graph[cur][edge[3]]
				rev_edge[1] += f
				cur = u
			
			if total_flow >= 1:
				current_score = -total_cost
				if current_score > best_score:
					best_score = current_score
		
		return best_score