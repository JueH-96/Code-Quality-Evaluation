from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # The 3x3 grid contains four possible 2x2 subgrids.
        # These subgrids start at the top-left corners (0,0), (0,1), (1,0), and (1,1).
        
        # Iterate through the possible top-left corner coordinates (r_offset, c_offset)
        # for a 2x2 subgrid.
        for r_offset in range(2): # Row index for the top row of the 2x2 subgrid (0 or 1)
            for c_offset in range(2): # Column index for the left column of the 2x2 subgrid (0 or 1)
                
                # Count the number of 'B's and 'W's within the current 2x2 subgrid.
                # The four cells in this subgrid are:
                # grid[r_offset][c_offset]
                # grid[r_offset][c_offset + 1]
                # grid[r_offset + 1][c_offset]
                # grid[r_offset + 1][c_offset + 1]
                
                b_count = 0
                w_count = 0
                
                # Check each of the four cells in the 2x2 subgrid
                if grid[r_offset][c_offset] == 'B':
                    b_count += 1
                else: # grid[r_offset][c_offset] == 'W'
                    w_count += 1

                if grid[r_offset][c_offset + 1] == 'B':
                    b_count += 1
                else: # grid[r_offset][c_offset + 1] == 'W'
                    w_count += 1

                if grid[r_offset + 1][c_offset] == 'B':
                    b_count += 1
                else: # grid[r_offset + 1][c_offset] == 'W'
                    w_count += 1
                    
                if grid[r_offset + 1][c_offset + 1] == 'B':
                    b_count += 1
                else: # grid[r_offset + 1][c_offset + 1] == 'W'
                    w_count += 1

                # A 2x2 square can be made monochrome (all the same color)
                # by changing *at most one cell* if, within that 2x2 square,
                # the count of one color is 3 and the other is 1, OR
                # the count of one color is 4 and the other is 0.
                # This means the count of either 'B' or 'W' is 3 or more.
                # If b_count is 3 or 4 (w_count is 1 or 0), we can make it all 'B's with 0 or 1 change.
                # If w_count is 3 or 4 (b_count is 1 or 0), we can make it all 'W's with 0 or 1 change.
                if b_count >= 3 or w_count >= 3:
                    # If we find just one 2x2 subgrid that satisfies this condition,
                    # it is possible to make a 2x2 square of the same color
                    # by changing at most one cell in the overall grid (specifically,
                    # changing the single outlier cell within this subgrid if needed).
                    return True

        # If we have checked all four possible 2x2 subgrids and none of them
        # can be made monochrome with at most one change, then it's impossible
        # to satisfy the condition for the entire grid.
        return False