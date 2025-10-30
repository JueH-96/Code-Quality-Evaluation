from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Iterate through all possible top-left corners of 2x2 subgrids.
        # A 3x3 grid has 4 such 2x2 subgrids.
        # Their top-left corners can be (r_start, c_start) where r_start and c_start are 0 or 1.
        
        for r_start in range(2):  # r_start can be 0 or 1
            for c_start in range(2):  # c_start can be 0 or 1
                
                # Count 'W's and 'B's in the current 2x2 subgrid.
                # The subgrid cells are:
                # grid[r_start][c_start],     grid[r_start][c_start+1]
                # grid[r_start+1][c_start], grid[r_start+1][c_start+1]
                
                white_count = 0
                black_count = 0
                
                # Iterate over the 2x2 cells within this subgrid
                # r_idx will go from r_start to r_start+1
                # c_idx will go from c_start to c_start+1
                for r_idx in range(r_start, r_start + 2):
                    for c_idx in range(c_start, c_start + 2):
                        if grid[r_idx][c_idx] == 'W':
                            white_count += 1
                        else:  # grid[r_idx][c_idx] == 'B'
                            black_count += 1
                
                # Check if this subgrid can be made monochromatic with at most one change.
                # A 2x2 subgrid has 4 cells, so white_count + black_count == 4.
                
                # To make it all 'W':
                # Need to change all 'B's to 'W's.
                # Number of changes = black_count.
                # Possible if black_count <= 1 (i.e., 0 or 1 'B's).
                # If black_count == 0, it's already all 'W' (0 changes).
                # If black_count == 1, one 'B' needs to be changed (1 change).
                if black_count <= 1:
                    return True
                    
                # To make it all 'B':
                # Need to change all 'W's to 'B's.
                # Number of changes = white_count.
                # Possible if white_count <= 1 (i.e., 0 or 1 'W's).
                # If white_count == 0, it's already all 'B' (0 changes).
                # If white_count == 1, one 'W' needs to be changed (1 change).
                if white_count <= 1:
                    return True
                    
        # If none of the four 2x2 subgrids satisfy the condition after checking all of them.
        return False