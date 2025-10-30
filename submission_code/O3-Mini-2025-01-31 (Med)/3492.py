from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        
        # Create two prefix sum matrices for 'X' and 'Y'
        preX = [[0] * m for _ in range(n)]
        preY = [[0] * m for _ in range(n)]
        
        ans = 0
        
        for i in range(n):
            for j in range(m):
                # Count current cell
                curX = 1 if grid[i][j] == 'X' else 0
                curY = 1 if grid[i][j] == 'Y' else 0
                
                # Compute prefix sum for preX
                topX = preX[i-1][j] if i > 0 else 0
                leftX = preX[i][j-1] if j > 0 else 0
                diagX = preX[i-1][j-1] if i > 0 and j > 0 else 0
                preX[i][j] = curX + topX + leftX - diagX
                
                # Similarly for preY
                topY = preY[i-1][j] if i > 0 else 0
                leftY = preY[i][j-1] if j > 0 else 0
                diagY = preY[i-1][j-1] if i > 0 and j > 0 else 0
                preY[i][j] = curY + topY + leftY - diagY
                
                # Now, the submatrix defined by (0,0) to (i,j) has:
                countX = preX[i][j]
                countY = preY[i][j]
                
                # Check if this submatrix contains at least one 'X' and if X and Y frequency are equal.
                if countX > 0 and countX == countY:
                    ans += 1
        
        return ans