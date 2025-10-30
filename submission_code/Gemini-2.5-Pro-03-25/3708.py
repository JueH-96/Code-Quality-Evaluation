import math
from typing import List

class Solution:
    """
    Solves the problem of traversing a 2D grid in a zigzag pattern while skipping alternate cells.
    """
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        """
        Traverses the grid in a zigzag pattern, skipping every alternate cell, 
        and returns the values of the visited cells.

        Args:
            grid: An m x n 2D list of positive integers.

        Returns:
            A list of integers representing the values of the cells visited 
            during the zigzag traversal with skips.
        """
        
        if not grid or not grid[0]:
            return []  # Handle empty grid case, though constraints prevent this

        n = len(grid)  # Number of rows
        m = len(grid[0]) # Number of columns
        
        result = []
        # This boolean flag determines whether the current cell should be added to the result.
        # It starts as True because the first cell (0, 0) should always be included.
        should_include = True 
        
        # Iterate through each row
        for r in range(n):
            # Determine the direction of traversal based on the row index (even or odd)
            if r % 2 == 0:
                # Even row (0, 2, 4, ...): Traverse Left to Right
                for c in range(m):
                    if should_include:
                        result.append(grid[r][c])
                    # Flip the flag for the next cell
                    should_include = not should_include
            else:
                # Odd row (1, 3, 5, ...): Traverse Right to Left
                for c in range(m - 1, -1, -1): # Iterate from m-1 down to 0
                    if should_include:
                        result.append(grid[r][c])
                    # Flip the flag for the next cell
                    should_include = not should_include
                        
        return result