from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        visited = [[float('inf') for _ in range(n)] for _ in range(m)]
        visited[0][0] = grid[0][0]
        
        while heap:
            cost, i, j = heapq.heappop(heap)
            if i == m - 1 and j == n - 1:
                return cost <= health
            for dir in directions:
                ni, nj = i + dir[0], j + dir[1]
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = cost + grid[ni][nj]
                    if new_cost < visited[ni][nj]:
                        visited[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        return False