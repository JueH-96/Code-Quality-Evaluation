class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    for k in range(j+1, n):
                        if grid[i][k] == 1:
                            for l in range(i+1, m):
                                if grid[l][j] == 1 and grid[l][k] == 1:
                                    count += 1
        
        return count