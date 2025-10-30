from collections import deque
from typing import List

class Solution:
	def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
		n = len(edges1) + 1
		m = len(edges2) + 1
		
		d1 = self.get_diameter(n, edges1)
		d2 = self.get_diameter(m, edges2)
		
		r1 = (d1 + 1) // 2
		r2 = (d2 + 1) // 2
		
		return max(d1, d2, r1 + r2 + 1)
	
	def get_diameter(self, n: int, edges: List[List[int]]) -> int:
		if n == 1:
			return 0
		graph = [[] for _ in range(n)]
		for a, b in edges:
			graph[a].append(b)
			graph[b].append(a)
		
		dist0 = [-1] * n
		q = deque([0])
		dist0[0] = 0
		while q:
			u = q.popleft()
			for v in graph[u]:
				if dist0[v] == -1:
					dist0[v] = dist0[u] + 1
					q.append(v)
		
		u = 0
		for i in range(1, n):
			if dist0[i] > dist0[u]:
				u = i
		
		dist = [-1] * n
		q = deque([u])
		dist[u] = 0
		while q:
			node = q.popleft()
			for v in graph[node]:
				if dist[v] == -1:
					dist[v] = dist[node] + 1
					q.append(v)
		
		return max(dist)