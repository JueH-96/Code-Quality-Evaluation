class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # The grid is 3x3, so there are four possible 2x2 subgrids.
        # We check each of these subgrids by their top-left corner (r, c).
        for r in range(2):
            for c in range(2):
                # In a 2x2 subgrid, count the number of black ('B') cells.
                # Python's True/False values can be treated as 1/0 in arithmetic.
                b_count = (grid[r][c] == 'B') + \
                          (grid[r][c+1] == 'B') + \
                          (grid[r+1][c] == 'B') + \
                          (grid[r+1][c+1] == 'B')

                # A 2x2 square can be made monochromatic with at most one change
                # if it does not have an even 2-2 split of colors.
                # If b_count is 0 or 4, it's already monochromatic (0 changes).
                # If b_count is 1 or 3, it needs only one change.
                # If b_count is 2, it needs two changes, which is not allowed.
                if b_count != 2:
                    return True
        
        # If no subgrid can be made monochromatic after checking all four, return False.
        return False