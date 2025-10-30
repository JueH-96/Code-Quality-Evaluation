from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        INF = float('inf')
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        heap = []
        heapq.heappush(heap, (dist[0][0], 0, 0))
        
        while heap:
            current_sum, i, j = heapq.heappop(heap)
            if i == m - 1 and j == n - 1:
                return current_sum <= health - 1
            if current_sum > dist[i][j]:
                continue
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_sum = current_sum + grid[ni][nj]
                    if new_sum < dist[ni][nj]:
                        dist[ni][nj] = new_sum
                        heapq.heappush(heap, (new_sum, ni, nj))
        return False