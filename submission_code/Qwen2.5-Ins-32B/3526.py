from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def is_palindrome(row):
            return row == row[::-1]
        
        def flip_cells(grid, is_row):
            flips = 0
            for i in range(m if is_row else n):
                if is_row:
                    row = grid[i]
                else:
                    row = [grid[j][i] for j in range(m)]
                if not is_palindrome(row):
                    for j in range(n // 2):
                        if row[j] != row[n - j - 1]:
                            flips += 1
            return flips
        
        row_flips = flip_cells(grid, True)
        col_flips = flip_cells(grid, False)
        
        return min(row_flips, col_flips)