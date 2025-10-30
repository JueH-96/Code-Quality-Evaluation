from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        """
        Calculates the number of submatrices starting at (0,0) that have an equal
        number of 'X's and 'Y's, and at least one 'X'.
        """
        rows = len(grid)
        cols = len(grid[0])
        
        # prefix_X[r+1][c+1] will store the count of 'X's in the subgrid from (0,0) to (r,c).
        # Similarly for prefix_Y.
        # We use (rows+1)x(cols+1) matrices to simplify boundary checks with a padding of zeros.
        prefix_X = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_Y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        ans = 0
        
        # Iterate through each cell of the grid. Each cell (r, c) will be the bottom-right
        # corner of a submatrix that starts at (0, 0), thus containing grid[0][0].
        for r in range(rows):
            for c in range(cols):
                # Determine if the current cell contains an 'X' or 'Y'.
                is_X = 1 if grid[r][c] == 'X' else 0
                is_Y = 1 if grid[r][c] == 'Y' else 0
                
                # Calculate the total number of 'X's and 'Y's in the submatrix from (0,0) to (r,c)
                # using the 2D prefix sum recurrence relation.
                # count(0..r, 0..c) = val(r,c) + count(0..r-1, 0..c) + count(0..r, 0..c-1) - count(0..r-1, 0..c-1)
                current_X = prefix_X[r + 1][c] + prefix_X[r][c + 1] - prefix_X[r][c] + is_X
                current_Y = prefix_Y[r + 1][c] + prefix_Y[r][c + 1] - prefix_Y[r][c] + is_Y
                
                # Store these new cumulative counts in our prefix sum matrices.
                prefix_X[r + 1][c + 1] = current_X
                prefix_Y[r + 1][c + 1] = current_Y
                
                # A submatrix is valid if it meets the specified conditions:
                # 1. It has an equal number of 'X's and 'Y's.
                # 2. It contains at least one 'X'.
                if current_X > 0 and current_X == current_Y:
                    ans += 1
                    
        return ans