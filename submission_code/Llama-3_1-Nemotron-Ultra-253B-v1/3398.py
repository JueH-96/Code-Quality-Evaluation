from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def helper(g):
            squares = [
                [(0,0), (0,1), (1,0), (1,1)],
                [(0,1), (0,2), (1,1), (1,2)],
                [(1,0), (1,1), (2,0), (2,1)],
                [(1,1), (1,2), (2,1), (2,2)]
            ]
            for sq in squares:
                cells = [g[i][j] for (i,j) in sq]
                if all(c == 'B' for c in cells) or all(c == 'W' for c in cells):
                    return True
            return False
        
        if helper(grid):
            return True
        
        for i in range(3):
            for j in range(3):
                new_grid = [row.copy() for row in grid]
                new_grid[i][j] = 'B' if new_grid[i][j] == 'W' else 'W'
                if helper(new_grid):
                    return True
        
        return False