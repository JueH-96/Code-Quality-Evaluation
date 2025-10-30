class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        """
        We want to count the number of submatrices that:
          1) Contain cell (0, 0),
          2) Have an equal frequency of 'X' and 'Y',
          3) Have at least one 'X'.

        Observing that each valid submatrix must include (0, 0) as its top-left corner,
        we only need to consider submatrices defined by (0, 0) and (r, c) for all r, c.
        This yields O(R*C) possible submatrices (where R and C are the dimensions of grid).

        We use prefix sums for 'X' and 'Y'. Define:
          prefixX[i+1][j+1] = number of 'X's in submatrix (0,0)-(i,j)
          prefixY[i+1][j+1] = number of 'Y's in submatrix (0,0)-(i,j)

        Then the number of 'X's in (0,0)-(r,c) is prefixX[r+1][c+1],
        the number of 'Y's is prefixY[r+1][c+1].

        We check:
          - X_count = prefixX[r+1][c+1] > 0  (at least one 'X')
          - X_count == prefixY[r+1][c+1]     (equal frequency of 'X' and 'Y')

        The answer is the total count of such (r, c).
        """
        R, C = len(grid), len(grid[0])
        
        # Build prefix sums for X and Y
        prefixX = [[0]*(C+1) for _ in range(R+1)]
        prefixY = [[0]*(C+1) for _ in range(R+1)]
        
        for i in range(R):
            for j in range(C):
                prefixX[i+1][j+1] = prefixX[i][j+1] + prefixX[i+1][j] - prefixX[i][j] + (1 if grid[i][j] == 'X' else 0)
                prefixY[i+1][j+1] = prefixY[i][j+1] + prefixY[i+1][j] - prefixY[i][j] + (1 if grid[i][j] == 'Y' else 0)
        
        result = 0
        # Check every bottom-right corner (r, c)
        for r in range(R):
            for c in range(C):
                x_count = prefixX[r+1][c+1]
                y_count = prefixY[r+1][c+1]
                if x_count > 0 and x_count == y_count:
                    result += 1
        
        return result