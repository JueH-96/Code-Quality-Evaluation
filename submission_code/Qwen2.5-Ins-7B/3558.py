from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def is_safe(x, y, h):
            return 0 <= x < m and 0 <= y < n and (grid[x][y] == 0 or h - grid[x][y] > 0)
        
        queue = deque([(0, 0, health)])
        visited = set([(0, 0)])
        
        while queue:
            x, y, h = queue.popleft()
            if (x, y) == (m - 1, n - 1):
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and is_safe(nx, ny, h - grid[x][y]):
                    visited.add((nx, ny))
                    queue.append((nx, ny, h - grid[x][y]))
        
        return False