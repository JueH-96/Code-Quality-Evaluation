from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        
        # Because every submatrix must start at (0, 0), every valid submatrix is defined by its bottom-right coordinate (i, j)
        # In other words, the submatrix is grid[0:i+1][0:j+1].
        #
        # We need those submatrices with:
        #   1. Containing grid[0][0] (always the case)
        #   2. Equal frequency of 'X' and 'Y'
        #   3. At least one 'X' (and hence at least one 'Y' if they're equal)
        #
        # We can precompute two prefix sum matrices:
        #   prefX[i][j] = number of X's in submatrix grid[0:i+1][0:j+1]
        #   prefY[i][j] = number of Y's in submatrix grid[0:i+1][0:j+1]
        #
        # Then we will iterate over all possible (i, j) and count those submatrices that satisfy:
        #     prefX[i][j] == prefY[i][j] and prefX[i][j] > 0
        
        # Initialize prefix sum grids with extra row and column to avoid boundary complications.
        prefX = [[0]*(m+1) for _ in range(n+1)]
        prefY = [[0]*(m+1) for _ in range(n+1)]
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                # Determine current cell value counts.
                addX = 1 if grid[i-1][j-1] == 'X' else 0
                addY = 1 if grid[i-1][j-1] == 'Y' else 0
                # Compute prefix sums using inclusion-exclusion.
                prefX[i][j] = prefX[i-1][j] + prefX[i][j-1] - prefX[i-1][j-1] + addX
                prefY[i][j] = prefY[i-1][j] + prefY[i][j-1] - prefY[i-1][j-1] + addY
        
        ans = 0
        # Now consider all bottom-right corners.
        # The submatrix (0,0) to (i, j) corresponds to (i+1, j+1) in our prefix arrays.
        for i in range(1, n+1):
            for j in range(1, m+1):
                countX = prefX[i][j]
                countY = prefY[i][j]
                # Conditions: equal number of X and Y and at least one X.
                if countX == countY and countX > 0:
                    ans += 1
        return ans