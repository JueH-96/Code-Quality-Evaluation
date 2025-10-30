class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def is_palindromic(row):
            return row == row[::-1]
        
        def flip_cell(i, j):
            for x in range(m):
                grid[x][j] = 1 - grid[x][j]
            for y in range(n):
                grid[i][y] = 1 - grid[i][y]
        
        # Check if all rows are palindromic
        if all(is_palindromic(row) for row in grid):
            return 0
        
        # Check if all columns are palindromic
        if all(is_palindromic([grid[i][j] for i in range(m)]) for j in range(n)):
            return 0
        
        # Flip cells to make all rows or columns palindromic
        min_flips = float('inf')
        for i in range(m):
            for j in range(n):
                new_grid = [row[:] for row in grid]
                flip_cell(i, j)
                if all(is_palindromic(row) for row in new_grid) or all(is_palindromic([new_grid[x][y] for x in range(m)]) for y in range(n)):
                    min_flips = min(min_flips, 1)
                grid = new_grid
        
        return min_flips if min_flips != float('inf') else -1