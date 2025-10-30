from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        queue = deque([(0, 0, health)])
        visited[0][0] = True
        
        while queue:
            x, y, current_health = queue.popleft()
            
            if x == m - 1 and y == n - 1:
                return True
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                    new_health = current_health - grid[nx][ny]
                    
                    if new_health > 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny, new_health))
        
        return False