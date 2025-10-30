from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        heap = [(grid[0][0], 0, 0)]
        heapq.heapify(heap)
        
        while heap:
            cost, x, y = heapq.heappop(heap)
            if x == m - 1 and y == n - 1:
                break
            if cost > dist[x][y]:
                continue
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_cost = cost + grid[nx][ny]
                    if new_cost < dist[nx][ny]:
                        dist[nx][ny] = new_cost
                        heapq.heappush(heap, (new_cost, nx, ny))
        
        return dist[m-1][n-1] <= health - 1