import heapq
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0]) if m else 0
        INF = float('inf')
        distance = [[INF] * n for _ in range(m)]
        distance[0][0] = grid[0][0]
        heap = []
        heapq.heappush(heap, (distance[0][0], 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_sum, x, y = heapq.heappop(heap)
            if current_sum > distance[x][y]:
                continue
            if x == m - 1 and y == n - 1:
                break  # Early exit if we reach the end
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_sum = current_sum + grid[nx][ny]
                    if new_sum < distance[nx][ny]:
                        distance[nx][ny] = new_sum
                        heapq.heappush(heap, (new_sum, nx, ny))
        
        return distance[m-1][n-1] <= health - 1