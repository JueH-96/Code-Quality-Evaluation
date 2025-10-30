class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        rows_total = 0
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        # Calculate the flips needed for all rows to be palindromic
        for row in grid:
            length = len(row)
            count = 0
            for i in range(length // 2):
                if row[i] != row[length - 1 - i]:
                    count += 1
            rows_total += count
        
        cols_total = 0
        # Calculate the flips needed for all columns to be palindromic
        for j in range(n):
            count = 0
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    count += 1
            cols_total += count
        
        return min(rows_total, cols_total)