from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        row_flips = 0
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        # Calculate the number of flips needed for each row to be a palindrome
        for row in grid:
            length = len(row)
            for i in range(length // 2):
                if row[i] != row[length - 1 - i]:
                    row_flips += 1
        
        # Calculate the number of flips needed for each column to be a palindrome
        col_flips = 0
        for col in range(n):
            for i in range(m // 2):
                top = i
                bottom = m - 1 - i
                if grid[top][col] != grid[bottom][col]:
                    col_flips += 1
        
        return min(row_flips, col_flips)