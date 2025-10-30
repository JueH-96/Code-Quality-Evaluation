from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        count = 0

        # Iterate over each cell in the grid
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Check for right triangles with grid[i][j] as the right-angle vertex
                    for k in range(cols):
                        if k != j and grid[i][k] == 1:
                            for l in range(rows):
                                if l != i and grid[l][j] == 1:
                                    count += 1

        return count