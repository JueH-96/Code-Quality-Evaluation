from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # We will process diagonals identified by d = j - i
        for d in range(-(n - 1), n):
            # Collect the diagonal elements and their coordinates
            coords = []
            vals = []
            for i in range(n):
                j = i + d
                if 0 <= j < n:
                    coords.append((i, j))
                    vals.append(grid[i][j])
            # Decide sort order: diagonals with d <= 0 => non-increasing (reverse)
            # diagonals with d > 0 => non-decreasing
            if d <= 0:
                vals.sort(reverse=True)
            else:
                vals.sort()
            # Write back sorted values
            for (i, j), v in zip(coords, vals):
                grid[i][j] = v
        return grid