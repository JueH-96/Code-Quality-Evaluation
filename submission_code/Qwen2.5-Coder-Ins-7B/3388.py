class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        
        # Iterate over all pairs of 1s in the grid
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    for k in range(i + 1, rows):
                        if grid[k][j] == 1:
                            for l in range(cols):
                                if grid[i][l] == 1 and grid[k][l] == 1 and i != k and j != l:
                                    count += 1
        
        return count