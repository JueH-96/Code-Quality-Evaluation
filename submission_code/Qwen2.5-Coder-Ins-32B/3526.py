from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def flips_to_palindrome(row_or_col):
            flips = 0
            for i in range(len(row_or_col) // 2):
                if row_or_col[i] != row_or_col[-i - 1]:
                    flips += 1
            return flips
        
        # Calculate the cost to make all rows palindromic
        row_flips = 0
        for row in grid:
            row_flips += flips_to_palindrome(row)
        
        # Calculate the cost to make all columns palindromic
        col_flips = 0
        for col in range(n):
            column = [grid[row][col] for row in range(m)]
            col_flips += flips_to_palindrome(column)
        
        # Return the minimum of the two costs
        return min(row_flips, col_flips)