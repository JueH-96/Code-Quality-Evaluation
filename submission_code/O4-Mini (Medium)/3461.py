from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        """
        Find the smallest axis-aligned rectangle enclosing all 1's in the grid.
        Simply compute the bounding box of all 1-cells.
        """
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0

        # Initialize bounds
        min_i, max_i = m, -1
        min_j, max_j = n, -1

        # Scan the grid to update bounds
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if i < min_i:
                        min_i = i
                    if i > max_i:
                        max_i = i
                    if j < min_j:
                        min_j = j
                    if j > max_j:
                        max_j = j

        # The rectangle dimensions
        height = max_i - min_i + 1
        width = max_j - min_j + 1

        return height * width