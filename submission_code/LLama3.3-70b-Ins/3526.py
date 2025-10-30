from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Function to check if a row is palindromic
        def is_palindromic_row(row):
            return row == row[::-1]
        
        # Function to check if a column is palindromic
        def is_palindromic_col(col):
            return col == col[::-1]
        
        # Function to flip a cell
        def flip_cell(i, j):
            grid[i][j] = 1 - grid[i][j]
        
        # Check if all rows are palindromic
        rows_palindromic = all(is_palindromic_row(row) for row in grid)
        
        # If all rows are already palindromic, return 0
        if rows_palindromic:
            return 0
        
        # Check if all columns are palindromic
        cols_palindromic = all(is_palindromic_col([grid[i][j] for i in range(m)]) for j in range(n))
        
        # If all columns are already palindromic, return 0
        if cols_palindromic:
            return 0
        
        # Initialize minimum flips for rows and columns
        min_flips_rows = float('inf')
        min_flips_cols = float('inf')
        
        # Calculate minimum flips for rows
        for i in range(m):
            flips = 0
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - j - 1]:
                    flips += 1
            min_flips_rows = min(min_flips_rows, flips)
        
        # Calculate minimum flips for columns
        for j in range(n):
            flips = 0
            for i in range(m // 2):
                if grid[i][j] != grid[m - i - 1][j]:
                    flips += 1
            min_flips_cols = min(min_flips_cols, flips)
        
        # Return minimum of minimum flips for rows and columns
        return min(min_flips_rows, min_flips_cols)