from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        all_ones = (1 << n) - 1
        
        def flip_row(row):
            return row ^ all_ones
        
        def is_palindrome(row):
            return row == flip_row(row)
        
        def count_flips_to_palindrome(row):
            original = row
            row = flip_row(row)
            flips = 0
            while row != 0:
                bit = row & -row
                row -= bit
                flips += 1
            return flips + (original != row)
        
        row_flips = 0
        for row in grid:
            row_flips += count_flips_to_palindrome(row)
        
        col_flips = 0
        for col in range(n):
            col_val = 0
            for row in range(m):
                col_val |= grid[row][col] << (m - row - 1)
            col_flips += count_flips_to_palindrome(col_val)
        
        min_flips = min(row_flips, col_flips)
        return min_flips