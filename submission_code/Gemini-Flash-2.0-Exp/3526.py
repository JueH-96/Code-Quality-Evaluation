class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        ans = float('inf')
        
        # Check rows
        row_flips = 0
        for i in range(m):
            flips = 0
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    flips += 1
            row_flips += flips
        ans = min(ans, row_flips)
        
        # Check columns
        col_flips = 0
        for j in range(n):
            flips = 0
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    flips += 1
            col_flips += flips
        ans = min(ans, col_flips)
        
        return ans