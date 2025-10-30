import heapq
from typing import List

class Solution:
	def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
		m = len(grid)
		n = len(grid[0])
		if grid[0][0] >= health:
			return False
		
		dist = [[float('inf')] * n for _ in range(m)]
		dist[0][0] = grid[0][0]
		heap = []
		heapq.heappush(heap, (grid[0][0], 0, 0))
		
		while heap:
			k, i, j = heapq.heappop(heap)
			if k != dist[i][j]:
				continue
			if i == m - 1 and j == n - 1:
				return True
			for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
				ni, nj = i + di, j + dj
				if 0 <= ni < m and 0 <= nj < n:
					nk = k + grid[ni][nj]
					if nk >= health:
						continue
					if nk < dist[ni][nj]:
						dist[ni][nj] = nk
						heapq.heappush(heap, (nk, ni, nj))
		return False