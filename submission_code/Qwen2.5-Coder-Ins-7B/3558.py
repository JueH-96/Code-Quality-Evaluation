class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        def dfs(x, y, health):
            if x < 0 or x >= m or y < 0 or y >= n or health <= 0:
                return False
            if x == m - 1 and y == n - 1:
                return health >= 1
            grid[x][y] = -1  # Mark as visited
            health -= grid[x][y]
            if dfs(x + 1, y, health) or dfs(x - 1, y, health) or dfs(x, y + 1, health) or dfs(x, y - 1, health):
                return True
            health += grid[x][y]  # Backtrack
            grid[x][y] = 0  # Unmark
            return False
        
        return dfs(0, 0, health)