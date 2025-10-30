import heapq
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        INF = 10**9
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        heap = [(dist[0][0], 0, 0)]
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while heap:
            cost, i, j = heapq.heappop(heap)
            if cost != dist[i][j]:
                continue
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = cost + grid[ni][nj]
                    if new_cost < dist[ni][nj]:
                        dist[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        
        min_unsafe = dist[m-1][n-1]
        return health >= min_unsafe + 1