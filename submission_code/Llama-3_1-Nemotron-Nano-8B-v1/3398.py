from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def is_valid(g: List[List[str]], i: int, j: int) -> bool:
            val = g[i][j]
            return g[i][j+1] == val and g[i+1][j] == val and g[i+1][j+1] == val
        
        # Check all four possible 2x2 squares in the original grid
        for x in [0, 1]:
            for y in [0, 1]:
                if is_valid(grid, x, y):
                    return True
        
        # Check all possible single cell changes
        for i in range(3):
            for j in range(3):
                # Create a copy of the grid
                new_grid = [row.copy() for row in grid]
                # Flip the current cell
                new_grid[i][j] = 'W' if new_grid[i][j] == 'B' else 'B'
                # Check all four 2x2 squares in the new grid
                for x in [0, 1]:
                    for y in [0, 1]:
                        if is_valid(new_grid, x, y):
                            return True
        return False