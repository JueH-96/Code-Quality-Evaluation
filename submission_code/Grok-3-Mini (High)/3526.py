class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Calculate the number of flips needed to make all rows palindromic
        row_flips = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    row_flips += 1
        
        # Calculate the number of flips needed to make all columns palindromic
        col_flips = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1
        
        # Return the minimum of the two options
        return min(row_flips, col_flips)