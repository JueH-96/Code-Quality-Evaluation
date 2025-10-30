class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    for k in range(j+1, cols):
                        if grid[i][k] == 1 and grid[k][j] == 1:
                            count += 1

        return count