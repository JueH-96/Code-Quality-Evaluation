import heapq
from collections import deque, defaultdict
from typing import List

class Solution:
	def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
		n = len(edges) + 1
		graph = defaultdict(list)
		for u, v, w in edges:
			graph[u].append((v, w))
			graph[v].append((u, w))
		
		parent = [-1] * n
		children = [[] for _ in range(n)]
		q = deque([0])
		parent[0] = -1
		order = []
		while q:
			u = q.popleft()
			order.append(u)
			for v, w in graph[u]:
				if v == parent[u]:
					continue
				parent[v] = u
				children[u].append((v, w))
				q.append(v)
		
		dp0 = [0] * n
		dp1 = [0] * n
		
		for u in reversed(order):
			base_total = 0
			gains = []
			for v, w in children[u]:
				base_total += dp0[v]
				gain_val = w + dp1[v] - dp0[v]
				gains.append(gain_val)
			
			heap = []
			s = 0
			for g in gains:
				if g <= 0:
					continue
				if len(heap) < k:
					heapq.heappush(heap, g)
					s += g
				else:
					if g > heap[0]:
						removed = heapq.heappop(heap)
						s -= removed
						heapq.heappush(heap, g)
						s += g
			
			dp0[u] = base_total + s
			
			if k == 0:
				top_k_minus_1 = 0
			else:
				if len(heap) <= k - 1:
					top_k_minus_1 = s
				else:
					top_k_minus_1 = s - heap[0]
			
			dp1[u] = base_total + top_k_minus_1
		
		return dp0[0]