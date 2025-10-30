class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Cost to make all rows palindromic
        cost_rows = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    cost_rows += 1
        
        # Cost to make all columns palindromic
        cost_cols = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    cost_cols += 1
        
        # Return the minimum of the two costs
        return min(cost_rows, cost_cols)