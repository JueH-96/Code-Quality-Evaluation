class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        count = 0
        for i in range(len(grid)):
            for j in range(i+1, len(grid)):
                if grid[i][j] == 1:
                    for k in range(j+1, len(grid[0])):
                        if grid[j][k] == 1:
                            count += 1
        return count