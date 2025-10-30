from collections import deque
from typing import List

class Solution:
	def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
		n = len(edges1) + 1
		m = len(edges2) + 1
		
		graph1 = [[] for _ in range(n)]
		for a, b in edges1:
			graph1[a].append(b)
			graph1[b].append(a)
		
		graph2 = [[] for _ in range(m)]
		for u, v in edges2:
			graph2[u].append(v)
			graph2[v].append(u)
		
		def get_diameter(graph, num_nodes):
			if num_nodes == 0:
				return 0
			dist1 = [-1] * num_nodes
			q = deque()
			q.append(0)
			dist1[0] = 0
			while q:
				node = q.popleft()
				for neighbor in graph[node]:
					if dist1[neighbor] == -1:
						dist1[neighbor] = dist1[node] + 1
						q.append(neighbor)
			u = 0
			for i in range(num_nodes):
				if dist1[i] > dist1[u]:
					u = i
			
			dist2 = [-1] * num_nodes
			q.append(u)
			dist2[u] = 0
			while q:
				node = q.popleft()
				for neighbor in graph[node]:
					if dist2[neighbor] == -1:
						dist2[neighbor] = dist2[node] + 1
						q.append(neighbor)
			v = u
			for i in range(num_nodes):
				if dist2[i] > dist2[v]:
					v = i
			return dist2[v]
		
		diam1 = get_diameter(graph1, n)
		diam2 = get_diameter(graph2, m)
		
		r1 = (diam1 + 1) // 2
		r2 = (diam2 + 1) // 2
		
		M = max(diam1, diam2)
		return max(M, 1 + r1 + r2)