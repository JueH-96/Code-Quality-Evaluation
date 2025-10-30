import collections
from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        """
        Checks if any 2x2 subgrid in the 3x3 grid can be made monochromatic 
        (all 'B' or all 'W') by changing the color of at most one cell.

        The grid is guaranteed to be 3x3 and contain only 'B' or 'W'.
        A 2x2 square can be made monochromatic with at most one change if the number 
        of 'B's (or 'W's) in that square is 0, 1, 3, or 4. This is equivalent to 
        checking if the count of 'B's is not equal to 2 (or count of 'W's is not 2).

        Args:
            grid: A 3x3 list of lists of strings, where each element is either 'B' or 'W'.

        Returns:
            True if it's possible to form such a 2x2 square by changing at most one cell, 
            False otherwise.
        """
        
        # The grid dimensions are fixed at 3x3 based on constraints.
        rows = 3
        cols = 3

        # There are four possible 2x2 subgrids in a 3x3 grid.
        # We iterate through the potential top-left corners (r, c) of these subgrids.
        # The top-left corner coordinates can be (0, 0), (0, 1), (1, 0), or (1, 1).
        # The loops range(rows - 1) and range(cols - 1) correctly cover these coordinates.
        for r in range(rows - 1):  # r iterates through 0, 1
            for c in range(cols - 1): # c iterates through 0, 1
                
                # For each potential 2x2 subgrid starting at top-left corner (r, c),
                # count the number of black ('B') cells.
                count_b = 0 # Initialize count of black cells for the current subgrid
                
                # The four cells of the 2x2 subgrid are:
                # (r, c),   (r, c+1)
                # (r+1, c), (r+1, c+1)
                
                # Check the color of each cell in the 2x2 subgrid.
                if grid[r][c] == 'B':
                    count_b += 1
                if grid[r][c+1] == 'B':
                    count_b += 1
                if grid[r+1][c] == 'B':
                    count_b += 1
                if grid[r+1][c+1] == 'B':
                    count_b += 1
                
                # The number of white ('W') cells in the 2x2 subgrid is 4 - count_b.
                # Let count_w = 4 - count_b.
                
                # Determine if this 2x2 subgrid can be made monochromatic with at most one change.
                # This requires 0 or 1 change.
                # - 0 changes needed: If the square is already monochromatic (all 'B' or all 'W').
                #   This occurs when count_b = 4 or count_b = 0 (which means count_w = 4).
                # - 1 change needed: If the square has 3 cells of one color and 1 of the other.
                #   This occurs when count_b = 3 (change the one 'W' to 'B') or 
                #   count_b = 1 (change the one 'B' to 'W').
                
                # Combining these conditions, the goal is achievable if count_b is 0, 1, 3, or 4.
                # This is logically equivalent to the condition that count_b is not equal to 2.
                # If count_b = 2, then count_w must also be 2. To make the square monochromatic,
                # we would need to change 2 cells (either the 2 'B's or the 2 'W's). 
                # This requires 2 changes, which is more than the allowed limit of 1.
                
                if count_b != 2:
                    # If the count of black cells is not 2, it means this 2x2 subgrid 
                    # either is already monochromatic or can be made so with one change.
                    # Since we found a valid subgrid, we can immediately return True.
                    return True

        # If we have iterated through all four possible 2x2 subgrids and none of them 
        # satisfied the condition (i.e., for all subgrids, count_b was 2), 
        # then it's impossible to achieve the goal.
        return False