import heapq
from collections import deque
from typing import List

class Solution:
	def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
		n = len(edges) + 1
		graph = [[] for _ in range(n)]
		for e in edges:
			u, v, w = e
			graph[u].append((v, w))
			graph[v].append((u, w))
		
		parent = [-1] * n
		children = [[] for _ in range(n)]
		order = []
		q = deque([0])
		parent[0] = -1
		while q:
			u = q.popleft()
			order.append(u)
			for v, w in graph[u]:
				if v == parent[u]:
					continue
				parent[v] = u
				children[u].append((v, w))
				q.append(v)
				
		rev_order = order[::-1]
		f = [0] * n
		g = [0] * n
		
		for u in rev_order:
			base = 0
			heap = []
			total_gain = 0
			for v, w in children[u]:
				base += f[v]
				gain = w + g[v] - f[v]
				if gain <= 0:
					continue
				if len(heap) < k:
					heapq.heappush(heap, gain)
					total_gain += gain
				else:
					if gain > heap[0]:
						pop_val = heapq.heappop(heap)
						total_gain -= pop_val
						heapq.heappush(heap, gain)
						total_gain += gain
			f[u] = base + total_gain
			if len(heap) < k:
				g[u] = base + total_gain
			else:
				g[u] = base + total_gain - heap[0]
				
		return f[0]