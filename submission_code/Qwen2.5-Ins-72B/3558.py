from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = [[False] * n for _ in range(m)]
        
        def is_valid(x, y):
            return 0 <= x < m and 0 <= y < n and not visited[x][y]
        
        queue = deque([(0, 0, health - grid[0][0])])
        visited[0][0] = True
        
        while queue:
            x, y, current_health = queue.popleft()
            if current_health <= 0:
                continue
            if x == m - 1 and y == n - 1:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    visited[nx][ny] = True
                    queue.append((nx, ny, current_health - grid[nx][ny]))
        
        return False