class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Calculate the minimum flips to make all rows palindromic
        row_flips = 0
        for i in range(m):
            left = 0
            right = n - 1
            while left < right:
                if grid[i][left] != grid[i][right]:
                    row_flips += 1
                left += 1
                right -= 1
        
        # Calculate the minimum flips to make all columns palindromic
        col_flips = 0
        for j in range(n):
            top = 0
            bottom = m - 1
            while top < bottom:
                if grid[top][j] != grid[bottom][j]:
                    col_flips += 1
                top += 1
                bottom -= 1
        
        # Return the minimum of the two
        return min(row_flips, col_flips)