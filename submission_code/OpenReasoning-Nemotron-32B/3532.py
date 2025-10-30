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
		q = deque([0])
		parent[0] = -1
		while q:
			u = q.popleft()
			for v in graph[u]:
				if v == parent[u]:
					continue
				parent[v] = u
				children[u].append(v)
				q.append(v)
		
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
				w_uv = 1 if v % 2 == 1 else 2
				val = w_uv + down[v]
				if val > best1[u]:
					best2[u] = best1[u]
					best1[u] = val
				elif val > best2[u]:
					best2[u] = val
			if children[u]:
				down[u] = best1[u]
		
		up = [0] * n
		q = deque([0])
		while q:
			u = q.popleft()
			for v in children[u]:
				w_vu = 1 if u % 2 == 1 else 2
				w_uv = 1 if v % 2 == 1 else 2
				val_v = w_uv + down[v]
				if val_v == best1[u]:
					best_other = best2[u]
				else:
					best_other = best1[u]
				candidate = up[u]
				if best_other > candidate:
					candidate = best_other
				up[v] = w_vu + candidate
				q.append(v)
		
		return [max(down[i], up[i]) for i in range(n)]