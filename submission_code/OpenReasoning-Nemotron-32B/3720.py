from collections import deque
from typing import List

class Solution:
	def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
		if not edges:
			return -1
		max_w = max(w for _, _, w in edges)
		
		def feasible(X: int) -> bool:
			graph = [[] for _ in range(n)]
			for u, v, w in edges:
				if w <= X:
					graph[v].append(u)
			
			visited = [False] * n
			queue = deque([0])
			visited[0] = True
			count = 1
			while queue:
				cur = queue.popleft()
				for nxt in graph[cur]:
					if not visited[nxt]:
						visited[nxt] = True
						count += 1
						queue.append(nxt)
			return count == n
		
		low, high = 0, max_w + 1
		while low < high:
			mid = (low + high) // 2
			if feasible(mid):
				high = mid
			else:
				low = mid + 1
		return low if low <= max_w else -1