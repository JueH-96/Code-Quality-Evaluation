class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Check if a row is palindromic
        def is_palindromic_row(row):
            return row == row[::-1]
        
        # Check if a column is palindromic
        def is_palindromic_col(col):
            return col == col[::-1]
        
        # Count flips needed to make all rows palindromic
        def flip_rows():
            flips = 0
            for row in grid:
                if not is_palindromic_row(row):
                    flips += sum(row) != len(row) // 2
            return flips
        
        # Count flips needed to make all columns palindromic
        def flip_cols():
            flips = 0
            for col in zip(*grid):
                if not is_palindromic_col(col):
                    flips += sum(col) != len(col) // 2
            return flips
        
        return min(flip_rows(), flip_cols())