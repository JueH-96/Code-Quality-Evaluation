import heapq
from typing import List

class Solution:
	def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
		m = len(grid)
		n = len(grid[0])
		if m == 1 and n == 1:
			return health - grid[0][0] >= 1
		
		INF = 10**9
		dist = [[INF] * n for _ in range(m)]
		dist[0][0] = grid[0][0]
		
		if dist[0][0] < health:
			pq = [(dist[0][0], 0, 0)]
		else:
			pq = []
		
		directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		
		while pq:
			u, i, j = heapq.heappop(pq)
			if u != dist[i][j]:
				continue
			if i == m-1 and j == n-1:
				break
			for dx, dy in directions:
				ni, nj = i + dx, j + dy
				if 0 <= ni < m and 0 <= nj < n:
					new_unsafe = u + grid[ni][nj]
					if new_unsafe < health and new_unsafe < dist[ni][nj]:
						dist[ni][nj] = new_unsafe
						heapq.heappush(pq, (new_unsafe, ni, nj))
		
		return dist[m-1][n-1] <= health - 1