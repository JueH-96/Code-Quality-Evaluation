import heapq
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        INF = float('inf')
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        heap = []
        heapq.heappush(heap, (dist[0][0], 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_dist, i, j = heapq.heappop(heap)
            if i == m - 1 and j == n - 1:
                return current_dist <= (health - 1)
            if current_dist > dist[i][j]:
                continue
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n:
                    new_dist = current_dist + grid[ni][nj]
                    if new_dist < dist[ni][nj]:
                        dist[ni][nj] = new_dist
                        heapq.heappush(heap, (new_dist, ni, nj))
        return False