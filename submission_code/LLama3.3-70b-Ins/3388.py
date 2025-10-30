from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        # Iterate over all possible pairs of points
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Check for points in the same row
                    for k in range(cols):
                        if k != j and grid[i][k] == 1:
                            # Check for points in the same column
                            for x in range(rows):
                                if x != i and grid[x][j] == 1 and grid[x][k] == 1:
                                    count += 1
                    # Check for points in the same column
                    for k in range(rows):
                        if k != i and grid[k][j] == 1:
                            # Check for points in the same row
                            for x in range(cols):
                                if x != j and grid[i][x] == 1 and grid[k][x] == 1:
                                    count += 1
        
        # Divide by 2 to avoid counting each triangle twice
        return count // 2