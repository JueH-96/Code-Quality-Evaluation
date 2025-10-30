from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        if health <= 0:
            return False
        
        # Directions for moving in the grid (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS queue
        queue = deque([(0, 0, health)])  # (row, col, remaining health)
        visited = set((0, 0, health))
        
        while queue:
            x, y, h = queue.popleft()
            
            # If we reach the bottom-right corner
            if x == m - 1 and y == n - 1:
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    # Calculate new health after moving to the next cell
                    new_health = h - grid[nx][ny]
                    if new_health > 0 and (nx, ny, new_health) not in visited:
                        visited.add((nx, ny, new_health))
                        queue.append((nx, ny, new_health))
        
        return False