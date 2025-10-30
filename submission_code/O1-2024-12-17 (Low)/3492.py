class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n, m = len(grid), len(grid[0])

        # Prefix sums for X and Y:
        # pX[r][c] will store the count of 'X' in the submatrix from (0,0) to (r,c).
        # pY[r][c] will store the count of 'Y' in the submatrix from (0,0) to (r,c).
        pX = [[0]*m for _ in range(n)]
        pY = [[0]*m for _ in range(n)]
        
        # Fill prefix sums
        for r in range(n):
            for c in range(m):
                x_here = 1 if grid[r][c] == 'X' else 0
                y_here = 1 if grid[r][c] == 'Y' else 0
                if r == 0 and c == 0:
                    pX[r][c] = x_here
                    pY[r][c] = y_here
                elif r == 0:
                    pX[r][c] = pX[r][c-1] + x_here
                    pY[r][c] = pY[r][c-1] + y_here
                elif c == 0:
                    pX[r][c] = pX[r-1][c] + x_here
                    pY[r][c] = pY[r-1][c] + y_here
                else:
                    pX[r][c] = pX[r-1][c] + pX[r][c-1] - pX[r-1][c-1] + x_here
                    pY[r][c] = pY[r-1][c] + pY[r][c-1] - pY[r-1][c-1] + y_here
        
        # Count how many submatrices (0,0) to (r,c) have equal # of X and Y and at least one X
        result = 0
        for r in range(n):
            for c in range(m):
                if pX[r][c] == pY[r][c] and pX[r][c] > 0:
                    result += 1
        
        return result