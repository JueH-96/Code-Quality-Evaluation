from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        # According to constraints:
        # 1 <= grid.length (rows)
        # 1 <= grid[i].length (cols)
        # So, rows >= 1 and grid[0] is guaranteed to exist.
        cols = len(grid[0])

        # Initialize boundary coordinates.
        # Since we are guaranteed at least one '1' in the grid (problem constraint),
        # these initial values will be correctly updated.
        
        # min_r will store the smallest row index of a '1'.
        # Initialize to a value conceptually larger than any possible row index.
        min_r = rows 
        # max_r will store the largest row index of a '1'.
        # Initialize to a value conceptually smaller than any possible row index.
        max_r = -1   
        
        # Similarly for column indices.
        min_c = cols 
        max_c = -1   

        # Iterate through the grid to find the bounding box of all '1's.
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # Update minimum row index if current '1' is higher up.
                    min_r = min(min_r, r)
                    # Update maximum row index if current '1' is lower down.
                    max_r = max(max_r, r)
                    # Update minimum column index if current '1' is further left.
                    min_c = min(min_c, c)
                    # Update maximum column index if current '1' is further right.
                    max_c = max(max_c, c)
        
        # Calculate the height of the bounding rectangle.
        # The coordinates are 0-indexed. For example, if all '1's are in row 0,
        # min_r = 0, max_r = 0, so height = (0 - 0) + 1 = 1.
        height = max_r - min_r + 1
        
        # Calculate the width of the bounding rectangle.
        width = max_c - min_c + 1
        
        # The area is height multiplied by width.
        return height * width