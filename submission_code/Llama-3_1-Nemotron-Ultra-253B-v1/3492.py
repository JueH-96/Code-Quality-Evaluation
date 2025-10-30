from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        prefixX = [[0] * cols for _ in range(rows)]
        prefixY = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                currentX = 1 if grid[i][j] == 'X' else 0
                currentY = 1 if grid[i][j] == 'Y' else 0
                if i == 0 and j == 0:
                    prefixX[i][j] = currentX
                    prefixY[i][j] = currentY
                elif i == 0:
                    prefixX[i][j] = prefixX[i][j-1] + currentX
                    prefixY[i][j] = prefixY[i][j-1] + currentY
                elif j == 0:
                    prefixX[i][j] = prefixX[i-1][j] + currentX
                    prefixY[i][j] = prefixY[i-1][j] + currentY
                else:
                    prefixX[i][j] = prefixX[i-1][j] + prefixX[i][j-1] - prefixX[i-1][j-1] + currentX
                    prefixY[i][j] = prefixY[i-1][j] + prefixY[i][j-1] - prefixY[i-1][j-1] + currentY
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                x = prefixX[i][j]
                y = prefixY[i][j]
                if x == y and x >= 1:
                    count += 1
        return count