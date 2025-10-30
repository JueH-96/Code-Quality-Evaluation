from collections import deque
from typing import List

class Solution:
	def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
		max_val = 10**6
		low, high = 1, max_val
		ans = -1
		
		def check(X):
			graph = [[] for _ in range(n)]
			for u, v, w in edges:
				if w <= X:
					graph[v].append(u)
			visited = [False] * n
			q = deque([0])
			visited[0] = True
			count = 1
			while q:
				u = q.popleft()
				for v in graph[u]:
					if not visited[v]:
						visited[v] = True
						count += 1
						q.append(v)
			return count == n
		
		while low <= high:
			mid = (low + high) // 2
			if check(mid):
				ans = mid
				high = mid - 1
			else:
				low = mid + 1
				
		return ans