from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        count = 0
        
        # prev_prefix_x[c] stores the sum of 'X's in the rectangle from (0, 0) to (r-1, c)
        prev_prefix_x = [0] * n
        # prev_prefix_y[c] stores the sum of 'Y's in the rectangle from (0, 0) to (r-1, c)
        prev_prefix_y = [0] * n
        
        for r in range(m):
            # curr_row_px_1D stores the sum of 'X's in the current row r from column 0 to c
            curr_row_px_1D = 0
            # curr_row_py_1D stores the sum of 'Y's in the current row r from column 0 to c
            curr_row_py_1D = 0
            
            # curr_prefix_x[c] will store the sum of 'X's in the rectangle from (0, 0) to (r, c)
            curr_prefix_x = [0] * n
            # curr_prefix_y[c] will store the sum of 'Y's in the rectangle from (0, 0) to (r, c)
            curr_prefix_y = [0] * n
            
            for c in range(n):
                # Update 1D prefix sums for the current row
                if grid[r][c] == 'X':
                    curr_row_px_1D += 1
                elif grid[r][c] == 'Y':
                    curr_row_py_1D += 1
                    
                # Calculate 2D prefix sums up to (r, c)
                # The number of X's in the rectangle (0,0) to (r,c) is
                # (number of X's in (0,0) to (r-1, c)) + (number of X's in row r from 0 to c)
                # This formula is derived from:
                # Sum(0,0 to r,c) = Sum(0,0 to r-1, c) + Sum(r, 0 to c)
                curr_prefix_x[c] = prev_prefix_x[c] + curr_row_px_1D
                curr_prefix_y[c] = prev_prefix_y[c] + curr_row_py_1D
                
                # Check conditions for the submatrix with top-left (0, 0) and bottom-right (r, c)
                # 1. Contains grid[0][0] - This is implicitly satisfied by iterating submatrices starting at (0,0).
                # 2. Equal frequency of 'X' and 'Y'.
                # 3. At least one 'X'.
                if curr_prefix_x[c] > 0 and curr_prefix_x[c] == curr_prefix_y[c]:
                    count += 1
            
            # The 2D prefix sums calculated for the current row (r) become
            # the previous row's sums for the next iteration (r+1).
            prev_prefix_x = curr_prefix_x
            prev_prefix_y = curr_prefix_y
            
        return count