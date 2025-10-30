from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        # We want to count all submatrices that include (0,0)
        # and have equal number of 'X' and 'Y', with at least one 'X'.
        # Any submatrix that includes (0,0) must have its top-left corner at (0,0),
        # so it's simply the rectangle from (0,0) to (i,j) for each i,j.
        #
        # We'll build prefix sums for 'X' and 'Y' independently,
        # then for each (i,j) check if sumX(0..i,0..j) == sumY(0..i,0..j) >= 1.
        
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        # prefix sums with an extra row & col at front for easy calc
        psX = [[0]*(m+1) for _ in range(n+1)]
        psY = [[0]*(m+1) for _ in range(n+1)]
        
        for i in range(n):
            for j in range(m):
                psX[i+1][j+1] = psX[i+1][j] + psX[i][j+1] - psX[i][j] + (1 if grid[i][j]=='X' else 0)
                psY[i+1][j+1] = psY[i+1][j] + psY[i][j+1] - psY[i][j] + (1 if grid[i][j]=='Y' else 0)
        
        ans = 0
        # for every bottom-right corner (i,j) of a submatrix starting at (0,0)
        for i in range(n):
            for j in range(m):
                x_count = psX[i+1][j+1]
                y_count = psY[i+1][j+1]
                if x_count > 0 and x_count == y_count:
                    ans += 1
        
        return ans