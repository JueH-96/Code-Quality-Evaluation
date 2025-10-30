class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        row_flips = 0
        for row in grid:
            n = len(row)
            for i in range(n // 2):
                if row[i] != row[n - 1 - i]:
                    row_flips += 1
        
        column_flips = 0
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    column_flips += 1
        
        return min(row_flips, column_flips)