import heapq
from typing import List

class Solution:
	def minTimeToReach(self, moveTime: List[List[int]]) -> int:
		n = len(moveTime)
		m = len(moveTime[0])
		INF = 10**18
		dist = [[INF] * m for _ in range(n)]
		dist[0][0] = 0
		heap = [(0, 0, 0)]
		directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		
		while heap:
			t, i, j = heapq.heappop(heap)
			if i == n - 1 and j == m - 1:
				return t
			if t > dist[i][j]:
				continue
			for dx, dy in directions:
				ni, nj = i + dx, j + dy
				if 0 <= ni < n and 0 <= nj < m:
					nt = max(t, moveTime[ni][nj]) + 1
					if nt < dist[ni][nj]:
						dist[ni][nj] = nt
						heapq.heappush(heap, (nt, ni, nj))
		return dist[n-1][m-1]