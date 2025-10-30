class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        row_flips = 0
        for row in grid:
            for i in range(len(row) // 2):
                if row[i] != row[-i - 1]:
                    row_flips += 1
        
        col_flips = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1
        
        return min(row_flips, col_flips)