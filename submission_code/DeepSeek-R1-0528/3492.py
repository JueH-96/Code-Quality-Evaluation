class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        colX = [0] * n
        colY = [0] * n
        res = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'X':
                    colX[j] += 1
                elif grid[i][j] == 'Y':
                    colY[j] += 1
            
            runX = 0
            runY = 0
            for j in range(n):
                runX += colX[j]
                runY += colY[j]
                if runX == runY and runX > 0:
                    res += 1
        
        return res