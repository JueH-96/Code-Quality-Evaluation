from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        """
        Finds the smallest rectangle with horizontal and vertical sides
        that encloses all the 1's in the grid.
        """
        
        rows = len(grid)
        cols = len(grid[0])
        
        # Initialize boundaries to values outside the possible range of indices.
        # min values start high, max values start low.
        min_row = rows
        max_row = -1
        min_col = cols
        max_col = -1
        
        # Iterate through the grid to find the extremal coordinates of the '1's.
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # Update the top-most row found so far
                    min_row = min(min_row, r)
                    # Update the bottom-most row found so far
                    max_row = max(max_row, r)
                    # Update the left-most column found so far
                    min_col = min(min_col, c)
                    # Update the right-most column found so far
                    max_col = max(max_col, c)
                    
        # The problem constraints guarantee at least one '1', so the boundaries
        # will be valid indices from 0 to rows-1 and 0 to cols-1.
        
        # Calculate the height and width of the bounding rectangle.
        # The '+ 1' accounts for the inclusive nature of the boundary indices.
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # The area is the product of height and width.
        return height * width