from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def flip_row(row):
            return [1 - x for x in row]
        
        def is_palindromic(arr):
            return arr == arr[::-1]
        
        def make_rows_palindromic(grid):
            flips = 0
            for row in grid:
                if not is_palindromic(row):
                    flips += sum(1 for i in range(len(row) // 2) if row[i] != row[-i - 1])
            return flips
        
        def make_columns_palindromic(grid):
            flips = 0
            for col in range(len(grid[0])):
                column = [grid[row][col] for row in range(len(grid))]
                if not is_palindromic(column):
                    flips += sum(1 for i in range(len(column) // 2) if column[i] != column[-i - 1])
            return flips
        
        return min(make_rows_palindromic(grid), make_columns_palindromic(grid))