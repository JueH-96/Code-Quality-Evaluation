from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(x, y, health):
            if x == m - 1 and y == n - 1:
                return health - grid[x][y] >= 1
            
            if health <= grid[x][y]:
                return False
            
            original_value = grid[x][y]
            grid[x][y] = 2  # Mark as visited
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] != 2:
                    if dfs(nx, ny, health - original_value):
                        return True
            
            grid[x][y] = original_value  # Unmark as visited
            return False
        
        return dfs(0, 0, health)