from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # All possible top-left corners for 2x2 sub-squares in a 3x3 grid
        sub_squares = [
            [(0,0), (0,1), (1,0), (1,1)],
            [(0,1), (0,2), (1,1), (1,2)],
            [(1,0), (1,1), (2,0), (2,1)],
            [(1,1), (1,2), (2,1), (2,2)]
        ]
        
        # Function to check if a given 2x2 sub-square can be made
        # all color c ('B' or 'W') with at most one change
        def can_be_uniform_with_one_change(sq, c):
            diff = 0
            for (r, col) in sq:
                if grid[r][col] != c:
                    diff += 1
                    if diff > 1:
                        return False
            return True
        
        # Check each sub-square for both colors
        for sq in sub_squares:
            for color in ('B', 'W'):
                if can_be_uniform_with_one_change(sq, color):
                    return True
        
        return False