class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    for k in range(j + 1, n):
                        if grid[i][k]:
                            for l in range(i + 1, m):
                                if grid[l][j]:
                                    count += grid[l][k]
        return count