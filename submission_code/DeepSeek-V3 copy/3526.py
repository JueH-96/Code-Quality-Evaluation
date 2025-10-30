class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Function to calculate the minimum flips to make all rows palindromic
        def min_flips_rows():
            total_flips = 0
            for row in grid:
                left = 0
                right = n - 1
                flips = 0
                while left < right:
                    if row[left] != row[right]:
                        flips += 1
                    left += 1
                    right -= 1
                total_flips += flips
            return total_flips
        
        # Function to calculate the minimum flips to make all columns palindromic
        def min_flips_cols():
            total_flips = 0
            for col in range(n):
                top = 0
                bottom = m - 1
                flips = 0
                while top < bottom:
                    if grid[top][col] != grid[bottom][col]:
                        flips += 1
                    top += 1
                    bottom -= 1
                total_flips += flips
            return total_flips
        
        # Calculate the minimum flips for both options
        flips_rows = min_flips_rows()
        flips_cols = min_flips_cols()
        
        # Return the minimum of the two
        return min(flips_rows, flips_cols)