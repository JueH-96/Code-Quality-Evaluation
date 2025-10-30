class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        row_flips_all_rows = 0
        for i in range(m):
            row_flips = 0
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    row_flips += 1
            row_flips_all_rows += row_flips
            
        col_flips_all_cols = 0
        for j in range(n):
            col_flips = 0
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1
            col_flips_all_cols += col_flips
            
        return min(row_flips_all_rows, col_flips_all_cols)