from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Iterate through the possible top-left corners of a 2x2 square.
        # For a 3x3 grid, the top-left corners of 2x2 squares can be at:
        # (0,0), (0,1), (1,0), and (1,1).
        # We use range(2) for both rows and columns to cover indices 0 and 1.
        for r in range(2):  # Iterate over row indices: 0, 1
            for c in range(2):  # Iterate over column indices: 0, 1
                
                count_B = 0  # Counter for 'B' (Black) cells in the current 2x2 square
                count_W = 0  # Counter for 'W' (White) cells in the current 2x2 square
                
                # The four cells forming the current 2x2 square are:
                # (r, c), (r, c+1), (r+1, c), (r+1, c+1)
                
                # Check grid[r][c]
                if grid[r][c] == 'B':
                    count_B += 1
                else:
                    count_W += 1
                
                # Check grid[r][c+1]
                if grid[r][c+1] == 'B':
                    count_B += 1
                else:
                    count_W += 1
                    
                # Check grid[r+1][c]
                if grid[r+1][c] == 'B':
                    count_B += 1
                else:
                    count_W += 1
                    
                # Check grid[r+1][c+1]
                if grid[r+1][c+1] == 'B':
                    count_B += 1
                else:
                    count_W += 1
                
                # A 2x2 square can be made monochromatic with at most one change if:
                # - There are 3 or 4 'B's (meaning 0 or 1 'W's), so we change the 'W's to 'B's.
                #   This means count_B >= 3.
                # - OR there are 3 or 4 'W's (meaning 0 or 1 'B's), so we change the 'B's to 'W's.
                #   This means count_W >= 3.
                if count_B >= 3 or count_W >= 3:
                    return True  # We found such a 2x2 square, so return True
                    
        # If the loops complete, it means no 2x2 square satisfied the condition
        # (i.e., all 2x2 squares had 2 'B's and 2 'W's).
        # In this case, it's not possible to make a monochromatic 2x2 square with at most one change.
        return False