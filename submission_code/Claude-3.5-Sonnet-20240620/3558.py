class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        def dfs(i: int, j: int, h: int) -> bool:
            if i < 0 or i >= m or j < 0 or j >= n or h <= 0:
                return False
            
            if i == m - 1 and j == n - 1:
                return h > 0
            
            if grid[i][j] == 1:
                h -= 1
            
            grid[i][j] = 2  # Mark as visited
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if dfs(ni, nj, h):
                    return True
            
            grid[i][j] = 0  # Backtrack
            return False
        
        return dfs(0, 0, health)