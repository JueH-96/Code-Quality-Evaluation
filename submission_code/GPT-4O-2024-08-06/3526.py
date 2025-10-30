class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def min_flips_to_palindrome(arr):
            n = len(arr)
            flips = 0
            for i in range(n // 2):
                if arr[i] != arr[n - i - 1]:
                    flips += 1
            return flips
        
        m, n = len(grid), len(grid[0])
        
        # Calculate flips needed for each row to be palindromic
        row_flips = 0
        for row in grid:
            row_flips += min_flips_to_palindrome(row)
        
        # Calculate flips needed for each column to be palindromic
        col_flips = 0
        for j in range(n):
            col = [grid[i][j] for i in range(m)]
            col_flips += min_flips_to_palindrome(col)
        
        # Return the minimum of row_flips and col_flips
        return min(row_flips, col_flips)