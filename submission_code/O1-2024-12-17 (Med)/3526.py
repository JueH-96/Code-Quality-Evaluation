class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m else 0
        
        # Cost to make all rows palindromic
        row_cost = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    row_cost += 1
        
        # Cost to make all columns palindromic
        col_cost = 0
        for c in range(n):
            for i in range(m // 2):
                if grid[i][c] != grid[m - 1 - i][c]:
                    col_cost += 1
        
        return min(row_cost, col_cost)