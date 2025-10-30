from collections import deque
from typing import List

class Solution:
	def timeTaken(self, edges: List[List[int]]) -> List[int]:
		n = len(edges) + 1
		graph = [[] for _ in range(n)]
		for u, v in edges:
			graph[u].append(v)
			graph[v].append(u)
		
		parent = [-1] * n
		children = [[] for _ in range(n)]
		stack = [0]
		parent[0] = -1
		while stack:
			u = stack.pop()
			for v in graph[u]:
				if v == parent[u]:
					continue
				parent[v] = u
				children[u].append(v)
				stack.append(v)
		
		down = [0] * n
		best1 = [-10**18] * n
		best2 = [-10**18] * n
		
		stack = [0]
		order = []
		while stack:
			u = stack.pop()
			order.append(u)
			for v in children[u]:
				stack.append(v)
		for u in order[::-1]:
			for v in children[u]:
				w = 1 if v % 2 == 1 else 2
				val = w + down[v]
				if val > best1[u]:
					best2[u] = best1[u]
					best1[u] = val
				elif val > best2[u]:
					best2[u] = val
		
		up = [0] * n
		q = deque([0])
		while q:
			u = q.popleft()
			for v in children[u]:
				w_val = 1 if v % 2 == 1 else 2
				value_v = w_val + down[v]
				if value_v == best1[u]:
					candidate_other = best2[u]
				else:
					candidate_other = best1[u]
				candidate = max(up[u], candidate_other)
				cost_vu = 1 if u % 2 == 1 else 2
				up[v] = cost_vu + candidate
				q.append(v)
				
		ans = [max(down[u], up[u]) for u in range(n)]
		return ans