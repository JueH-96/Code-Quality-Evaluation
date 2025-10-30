from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS initialization
        queue = deque([(0, 0, health)])
        visited = set((0, 0))
        
        while queue:
            x, y, h = queue.popleft()
            
            # If we reached the bottom-right corner with health >= 1
            if (x, y) == (m - 1, n - 1) and h >= 1:
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    new_health = h - grid[nx][ny]
                    if new_health > 0:
                        queue.append((nx, ny, new_health))
                        visited.add((nx, ny))
        
        return False