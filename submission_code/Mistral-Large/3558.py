from typing import List
from heapq import heappop, heappush

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Priority queue to store (remaining_health, x, y)
        pq = [(-health, 0, 0)]
        visited = set((0, 0))

        while pq:
            current_health, x, y = heappop(pq)
            current_health = -current_health

            if x == m - 1 and y == n - 1:
                if current_health > 0:
                    return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    new_health = current_health - grid[nx][ny]
                    if new_health > 0:
                        visited.add((nx, ny))
                        heappush(pq, (-new_health, nx, ny))

        return False