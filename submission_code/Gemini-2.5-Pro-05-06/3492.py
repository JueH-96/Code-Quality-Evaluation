from typing import List

class Solution:
  def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
    R = len(grid)
    C = len(grid[0])
    
    ans = 0
    
    # dp_prev_x[j+1] stores the count of 'X's in the rectangle from (0,0) to (r-1, j).
    # (Here, r is the current row index in the outer loop, so r-1 is the previous row).
    # dp_prev_x[k] corresponds to the prefix sum up to column k-1 of the grid.
    # Example: dp_prev_x[1] is count for the submatrix ending at (r-1, 0).
    # dp_prev_x[0] is a padding element, always 0, to simplify boundary conditions.
    # These arrays store the prefix sums for the row *above* the current processing row `r`.
    dp_prev_x = [0] * (C + 1)
    dp_prev_y = [0] * (C + 1)

    # Iterate row by row (r is 0-indexed for grid access, from 0 to R-1)
    for r in range(R):
        # dp_curr_x[j+1] will store the count of 'X's in the rectangle from (0,0) to (r, j).
        # dp_curr_x[k] corresponds to the prefix sum up to column k-1 of the grid, for submatrices ending in current row r.
        dp_curr_x = [0] * (C + 1)
        dp_curr_y = [0] * (C + 1)
        
        # Iterate column by column (c is 0-indexed for grid access, from 0 to C-1)
        for c in range(C):
            is_x = 1 if grid[r][c] == 'X' else 0
            is_y = 1 if grid[r][c] == 'Y' else 0
            
            # Standard 2D prefix sum formula:
            # Let PS[i][j] be the sum of elements in the rectangle from (0,0) to (i,j).
            # PS[i][j] = PS[i-1][j] + PS[i][j-1] - PS[i-1][j-1] + value_at_grid[i][j]
            #
            # Mapping to our 1D DP arrays (which are 1-indexed for columns, C+1 length):
            # dp_curr_x[c+1] (corresponds to PS[r][c] using 0-indexed r,c for grid)
            #   = dp_prev_x[c+1] (corresponds to PS[r-1][c])
            #   + dp_curr_x[c]   (corresponds to PS[r][c-1])
            #   - dp_prev_x[c]   (corresponds to PS[r-1][c-1])
            #   + is_x           (value of grid[r][c] for 'X' count)
            
            dp_curr_x[c+1] = dp_prev_x[c+1] + dp_curr_x[c] - dp_prev_x[c] + is_x
            dp_curr_y[c+1] = dp_prev_y[c+1] + dp_curr_y[c] - dp_prev_y[c] + is_y
            
            # num_x and num_y are total counts of 'X' and 'Y' respectively
            # in the submatrix from grid[0][0] to grid[r][c].
            num_x = dp_curr_x[c+1]
            num_y = dp_curr_y[c+1]
            
            # Check conditions for a valid submatrix:
            # 1. Contains grid[0][0] (implicit by iterating (r,c) as bottom-right of submatrix anchored at (0,0))
            # 2. Equal frequency of 'X' and 'Y' (num_x == num_y)
            # 3. At least one 'X' (num_x > 0, which is equivalent to num_x >= 1 for non-negative integers)
            if num_x > 0 and num_x == num_y:
                ans += 1
        
        # For the next row iteration, the current row's sums (dp_curr_x, dp_curr_y)
        # will become the "previous row's sums".
        dp_prev_x = dp_curr_x
        dp_prev_y = dp_curr_y
            
    return ans