class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        count = 0
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    for k in range(rows):
                        for l in range(cols):
                            if grid[k][l] == 1 and (i == k or j == l) and (i,j) != (k,l):
                                for m in range(rows):
                                    for n in range(cols):
                                        if grid[m][n] == 1 and (i == m or j == n) and (k == m or l == n) and (i,j) != (m,n) and (k,l) != (m,n) and (i!=k or j!=l) and (i!=m or j!=n) and (k!=m or l!=n):
                                            
                                            count +=1

        return count //6